"""
AWS CDK Stack for Battery Smart Voicebot.
Defines all infrastructure components.
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticache as elasticache,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class VoicebotStack(Stack):
    """Main infrastructure stack for Battery Smart Voicebot."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # VPC for network isolation
        self.vpc = self._create_vpc()

        # ECR Repository for container images
        self.ecr_repo = self._create_ecr_repository()

        # S3 buckets for audio storage
        self.audio_bucket = self._create_audio_bucket()

        # DynamoDB tables
        self.conversations_table = self._create_conversations_table()
        self.handoffs_table = self._create_handoffs_table()

        # ElastiCache Redis for sessions
        self.redis = self._create_redis_cluster()

        # ECS Fargate service
        self.fargate_service = self._create_fargate_service()

        # IAM policies for Bedrock, Polly, Transcribe
        self._attach_aws_service_policies()

    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with public and private subnets."""
        return ec2.Vpc(
            self,
            "VoicebotVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                ),
            ],
        )

    def _create_ecr_repository(self) -> ecr.Repository:
        """Create ECR repository for voicebot container images."""
        return ecr.Repository(
            self,
            "VoicebotRepo",
            repository_name="battery-smart-voicebot",
            removal_policy=RemovalPolicy.RETAIN,
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep last 10 images",
                    max_image_count=10,
                    rule_priority=1,
                ),
            ],
        )

    def _create_audio_bucket(self) -> s3.Bucket:
        """Create S3 bucket for audio recordings."""
        return s3.Bucket(
            self,
            "AudioBucket",
            bucket_name=f"battery-smart-voicebot-audio-{self.account}",
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="AudioRetention",
                    expiration=Duration.days(90),  # Keep audio for 90 days
                ),
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT],
                    allowed_origins=["*"],
                    allowed_headers=["*"],
                ),
            ],
        )

    def _create_conversations_table(self) -> dynamodb.Table:
        """Create DynamoDB table for conversation history."""
        table = dynamodb.Table(
            self,
            "ConversationsTable",
            table_name="voicebot-conversations",
            partition_key=dynamodb.Attribute(
                name="call_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True,
            time_to_live_attribute="ttl",
        )

        # GSI for driver queries
        table.add_global_secondary_index(
            index_name="driver-index",
            partition_key=dynamodb.Attribute(
                name="driver_phone",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            ),
        )

        return table

    def _create_handoffs_table(self) -> dynamodb.Table:
        """Create DynamoDB table for handoff alerts."""
        table = dynamodb.Table(
            self,
            "HandoffsTable",
            table_name="voicebot-handoffs",
            partition_key=dynamodb.Attribute(
                name="alert_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,  # For real-time updates
        )

        # GSI for status queries
        table.add_global_secondary_index(
            index_name="status-index",
            partition_key=dynamodb.Attribute(
                name="status",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="created_at",
                type=dynamodb.AttributeType.STRING
            ),
        )

        return table

    def _create_redis_cluster(self) -> elasticache.CfnCacheCluster:
        """Create ElastiCache Redis cluster for session management."""
        # Security group for Redis
        redis_sg = ec2.SecurityGroup(
            self,
            "RedisSG",
            vpc=self.vpc,
            description="Security group for Redis cluster",
            allow_all_outbound=True,
        )

        # Allow inbound from VPC
        redis_sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(self.vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(6379),
            description="Redis from VPC"
        )

        # Subnet group
        subnet_group = elasticache.CfnSubnetGroup(
            self,
            "RedisSubnetGroup",
            description="Subnet group for Redis",
            subnet_ids=[
                subnet.subnet_id
                for subnet in self.vpc.private_subnets
            ],
        )

        # Redis cluster
        return elasticache.CfnCacheCluster(
            self,
            "RedisCluster",
            cache_node_type="cache.t3.micro",
            engine="redis",
            num_cache_nodes=1,
            cache_subnet_group_name=subnet_group.ref,
            vpc_security_group_ids=[redis_sg.security_group_id],
            auto_minor_version_upgrade=True,
        )

    def _create_fargate_service(self) -> ecs_patterns.ApplicationLoadBalancedFargateService:
        """Create ECS Fargate service for the voicebot API."""
        # ECS Cluster
        cluster = ecs.Cluster(
            self,
            "VoicebotCluster",
            vpc=self.vpc,
            container_insights=True,
        )

        # Task definition
        task_definition = ecs.FargateTaskDefinition(
            self,
            "VoicebotTask",
            memory_limit_mib=2048,
            cpu=1024,
        )

        # Container
        container = task_definition.add_container(
            "VoicebotContainer",
            image=ecs.ContainerImage.from_ecr_repository(
                self.ecr_repo,
                tag="latest"
            ),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="voicebot",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={
                "ENVIRONMENT": "production",
                "AWS_REGION": self.region,
                "AUDIO_BUCKET": self.audio_bucket.bucket_name,
                "CONVERSATIONS_TABLE": self.conversations_table.table_name,
                "HANDOFFS_TABLE": self.handoffs_table.table_name,
            },
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
            ),
        )

        container.add_port_mappings(
            ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP)
        )

        # Fargate service with ALB
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "VoicebotService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            public_load_balancer=True,
            listener_port=443,
            target_protocol=ecs.Protocol.TCP,
            health_check_grace_period=Duration.seconds(60),
        )

        # Auto-scaling
        scaling = service.service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10,
        )

        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        scaling.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        return service

    def _attach_aws_service_policies(self) -> None:
        """Attach IAM policies for AWS services."""
        task_role = self.fargate_service.task_definition.task_role

        # Bedrock access
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],
            )
        )

        # Polly access
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "polly:SynthesizeSpeech",
                ],
                resources=["*"],
            )
        )

        # Transcribe access
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "transcribe:StartStreamTranscription",
                    "transcribe:StartTranscriptionJob",
                    "transcribe:GetTranscriptionJob",
                ],
                resources=["*"],
            )
        )

        # S3 access for audio bucket
        self.audio_bucket.grant_read_write(task_role)

        # DynamoDB access
        self.conversations_table.grant_read_write_data(task_role)
        self.handoffs_table.grant_read_write_data(task_role)
