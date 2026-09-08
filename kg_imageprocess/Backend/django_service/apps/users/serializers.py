from rest_framework import serializers
from .models import User, UserProfile, BiomechanicalSession, FrameAnalysis


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['age', 'height_cm', 'weight_kg', 'injury_history', 'fitness_level']


class AthleteContextSerializer(serializers.ModelSerializer):
    """سریالایزر مخصوص ارائه Context به سرویس FastAPI و گراف دانش"""
    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'profile']


class FrameAnalysisSerializer(serializers.ModelSerializer):
    """سریالایزر ثبت و خواندن فریم‌های تحلیل‌شده توسط FastAPI"""
    class Meta:
        model = FrameAnalysis
        fields = ['id', 'session', 'frame_number', 'keypoints_data', 'joint_angles', 'created_at']


class BiomechanicalSessionSerializer(serializers.ModelSerializer):
    """سریالایزر نشست‌های بیومکانیکی به همراه فریم‌های مربوطه"""
    frames = FrameAnalysisSerializer(many=True, read_only=True)

    class Meta:
        model = BiomechanicalSession
        fields = ['id', 'user', 'exercise_type', 'status', 'notes', 'created_at', 'frames']