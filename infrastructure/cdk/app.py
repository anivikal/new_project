#!/usr/bin/env python3
"""
AWS CDK Application for Battery Smart Voicebot Infrastructure.
"""

import os

import aws_cdk as cdk

from stacks.voicebot_stack import VoicebotStack

app = cdk.App()

# Environment configuration
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION", "ap-south-1")
)

# Create stacks
VoicebotStack(
    app,
    "BatterySmartVoicebotStack",
    env=env,
    description="Battery Smart Multilingual Voicebot Infrastructure"
)

app.synth()
