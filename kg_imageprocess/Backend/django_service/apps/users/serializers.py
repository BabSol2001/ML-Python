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
            'id',
            'user',
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
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']


class UserCreateSerializer(serializers.ModelSerializer):
    """
    سریالایزر ساخت کاربر جدید به همراه ایجاد اتوماتیک پروفایل خالی
    """
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'password', 'first_name', 'last_name']

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data['email'],
            username=validated_data['username'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', '')
        )
        UserProfile.objects.create(user=user)
        return user


class AthleteContextSerializer(serializers.ModelSerializer):
    """
    سریالایزر ارائه Context کامل ورزشکار به سرویس FastAPI و گراف دانش (Knowledge Graph)
    """
    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'first_name', 'last_name', 'profile']


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