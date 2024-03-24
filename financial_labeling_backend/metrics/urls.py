from django.urls import path

from .views import (
    CategoryTypeDistribution,
    MemberProgressAPI,
    ProgressAPI,
    RelationTypeDistribution,
    SpanTypeDistribution,
    BoundingBoxTypeDistribution,
    SegmentationTypeDistribution,
)

urlpatterns = [
    path(route="progress", view=ProgressAPI.as_view(), name="progress"),
    path(route="member-progress", view=MemberProgressAPI.as_view(), name="member_progress"),
    path(route="category-distribution", view=CategoryTypeDistribution.as_view(), name="category_distribution"),
    path(route="relation-distribution", view=RelationTypeDistribution.as_view(), name="relation_distribution"),
    path(route="span-distribution", view=SpanTypeDistribution.as_view(), name="span_distribution"),
    path(route="boundingbox-distribution", view=BoundingBoxTypeDistribution.as_view(), name="boundingbox_distribution"),
    path(route="segments-distribution", view=SegmentationTypeDistribution.as_view(), name="segments_distribution"),
]
