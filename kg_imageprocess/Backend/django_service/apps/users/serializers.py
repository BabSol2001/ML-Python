from rest_framework import serializers
from .models import User, UserProfile, BiomechanicalSession, FrameAnalysis


class UserProfileSerializer(serializers.ModelSerializer):
    """
    سریالایزر جامع پروفایل کاربر شامل تمام متغیرهای بیومکانیکی، آنتروپومتری و تمرینی
    """
    age = serializers.ReadOnlyField()

    class Meta:
        model = UserProfile
        fields = [
            # ۱. دموگرافیک و آنتروپومتری
            'date_of_birth',
            'age',
            'gender',
            'height_cm',
            'weight_kg',
            'torso_length_cm',
            'femur_length_cm',
            'tibia_length_cm',
            
            # ۲. سوابق پزشکی، آسیب‌دیدگی و وضعیت‌های آناتومیک
            'injury_history',
            'medical_conditions',
            'postural_deviations',
            'baseline_flexibility_limits',
            'physiological_status',
            
            # ۳. سبک زندگی و شغل
            'occupational_lifestyle',
            
            # ۴. اهداف ورزشی و درمان
            'target_sport',
            'primary_goal',
            'therapy_details',
            
            # ۵. ترجیحات تمرینی و تجهیزات
            'preferred_training_style',
            'used_equipment_gear',
            'fitness_level',
            
            'created_at',
            'updated_at',
        ]


class AthleteContextSerializer(serializers.ModelSerializer):
    """
    سریالایزر ارائه Context کامل ورزشکار به سرویس FastAPI و گراف دانش (Knowledge Graph)
    """
    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'profile']


class FrameAnalysisSerializer(serializers.ModelSerializer):
    """
    سریالایزر ثبت و دریافت فریم‌های تحلیل‌شده توسط خط لوله FastAPI
    """
    class Meta:
        model = FrameAnalysis
        fields = ['id', 'session', 'frame_number', 'keypoints_data', 'joint_angles', 'created_at']


class BiomechanicalSessionSerializer(serializers.ModelSerializer):
    """
    سریالایزر نشست‌های بیومکانیکی شامل ساختار ست، الگوی تکرار، Tempo و فریم‌های مربوطه
    """
    frames = FrameAnalysisSerializer(many=True, read_only=True)

    class Meta:
        model = BiomechanicalSession
        fields = [
            'id', 
            'user', 
            'exercise_type', 
            'set_system', 
            'target_reps_scheme', 
            'rest_interval_seconds', 
            'tempo', 
            'status', 
            'notes', 
            'created_at', 
            'frames'
        ]