from rest_framework import serializers
from .models import Exercise, WorkoutPlan, WorkoutPlanItem, CompletedSessionSummary


class ExerciseSerializer(serializers.ModelSerializer):
    """
    سریالایزر بانک مرجع حرکات ورزشی و متغیرهای بیومکانیکی
    """
    class Meta:
        model = Exercise
        fields = [
            'id',
            'name',
            'slug',
            'description',
            'primary_muscles',
            'secondary_muscles',
            'kinematic_chain',
            'plane_of_motion',
            'difficulty',
            'target_joints_to_analyze',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class WorkoutPlanItemSerializer(serializers.ModelSerializer):
    """
    سریالایزر جزئیات اجرای هر حرکت در برنامه تمرینی
    """
    exercise_details = ExerciseSerializer(source='exercise', read_only=True)

    class Meta:
        model = WorkoutPlanItem
        fields = [
            'id',
            'plan',
            'exercise',
            'exercise_details',
            'order',
            'target_sets',
            'target_reps_scheme',
            'prescribed_rpe',
            'tempo',
            'rest_period_seconds',
            'notes',
        ]
        read_only_fields = ['id']


class WorkoutPlanSerializer(serializers.ModelSerializer):
    """
    سریالایزر برنامه‌های تمرینی شامل تمام حرکات و مشخصات تجویزی
    """
    items = WorkoutPlanItemSerializer(many=True, read_only=True)

    class Meta:
        model = WorkoutPlan
        fields = [
            'id',
            'user',
            'title',
            'description',
            'is_active',
            'start_date',
            'end_date',
            'items',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class CompletedSessionSummarySerializer(serializers.ModelSerializer):
    """
    سریالایزر دریافت و ارائه خلاصه عملکرد بیومکانیکی جلسات (بازگشتی از FastAPI)
    """
    exercise_name = serializers.ReadOnlyField(source='exercise.name')

    class Meta:
        model = CompletedSessionSummary
        fields = [
            'id',
            'user',
            'exercise',
            'exercise_name',
            'neo4j_session_uuid',
            'total_completed_reps',
            'valid_reps_count',
            'invalid_reps_count',
            'overall_form_score',
            'detected_compensations',
            'joint_range_of_motion_summary',
            'perceived_rpe',
            'ai_feedback_summary',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']