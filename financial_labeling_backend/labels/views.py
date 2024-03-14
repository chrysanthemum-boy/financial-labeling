from functools import partial
from typing import Type
import os
import requests
from django.core.exceptions import ValidationError
from django.shortcuts import get_object_or_404, get_list_or_404
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.http import HttpResponse

from .detect_auto import run_detect

from .permissions import CanEditLabel
from .serializers import (
    BoundingBoxSerializer,
    CategorySerializer,
    RelationSerializer,
    SegmentationSerializer,
    SpanSerializer,
    TextLabelSerializer,
)
from labels.models import (
    BoundingBox,
    Category,
    # DetectImage,
    Label,
    Relation,
    Segmentation,
    Span,
    TextLabel,
)
from projects.models import Project
from projects.permissions import IsProjectMember
from utils.models import Example

class BaseListAPI(generics.ListCreateAPIView):
    label_class: Type[Label]
    pagination_class = None
    permission_classes = [IsAuthenticated & IsProjectMember]
    swagger_schema = None

    @property
    def project(self):
        return get_object_or_404(Project, pk=self.kwargs["project_id"])

    def get_queryset(self):
        queryset = self.label_class.objects.filter(example=self.kwargs["example_id"])
        if not self.project.collaborative_annotation:
            queryset = queryset.filter(user=self.request.user)
        return queryset

    def create(self, request, *args, **kwargs):
        request.data["example"] = self.kwargs["example_id"]
        try:
            response = super().create(request, args, kwargs)
        except ValidationError as err:
            response = Response({"detail": err.messages}, status=status.HTTP_400_BAD_REQUEST)
        return response

    def perform_create(self, serializer):
        serializer.save(example_id=self.kwargs["example_id"], user=self.request.user)

    def delete(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        queryset.all().delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class BaseDetailAPI(generics.RetrieveUpdateDestroyAPIView):
    lookup_url_kwarg = "annotation_id"
    swagger_schema = None

    @property
    def project(self):
        return get_object_or_404(Project, pk=self.kwargs["project_id"])

    def get_permissions(self):
        if self.project.collaborative_annotation:
            self.permission_classes = [IsAuthenticated & IsProjectMember]
        else:
            self.permission_classes = [IsAuthenticated & IsProjectMember & partial(CanEditLabel, self.queryset)]
        return super().get_permissions()


class CategoryListAPI(BaseListAPI):
    label_class = Category
    serializer_class = CategorySerializer

    def create(self, request, *args, **kwargs):
        if self.project.single_class_classification:
            self.get_queryset().delete()
        return super().create(request, args, kwargs)


class CategoryDetailAPI(BaseDetailAPI):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer


class SpanListAPI(BaseListAPI):
    label_class = Span
    serializer_class = SpanSerializer


class SpanDetailAPI(BaseDetailAPI):
    queryset = Span.objects.all()
    serializer_class = SpanSerializer


class TextLabelListAPI(BaseListAPI):
    label_class = TextLabel
    serializer_class = TextLabelSerializer


class TextLabelDetailAPI(BaseDetailAPI):
    queryset = TextLabel.objects.all()
    serializer_class = TextLabelSerializer


class RelationList(BaseListAPI):
    label_class = Relation
    serializer_class = RelationSerializer


class RelationDetail(BaseDetailAPI):
    queryset = Relation.objects.all()
    serializer_class = RelationSerializer


class BoundingBoxListAPI(BaseListAPI):
    label_class = BoundingBox
    serializer_class = BoundingBoxSerializer
    
    def get_detect_data(self, request, *args, **kwargs):
        example_id = kwargs["example_id"]

        model_path = os.getcwd() + "/labels/auto_models/models/yanbao_paper30_CDLA-best.onnx"
        image_dir_path = os.getcwd() + "/media/"
        
        example = get_object_or_404(Example, id = example_id)
        file_name = str(example.filename)

        if BoundingBox.objects.filter(example_id = example_id):
            bboxes = get_list_or_404(BoundingBox, example_id = example_id)
            res = run_detect(model_path, image_dir_path + file_name, 0.3, 0.5)
            for i in range(len(res)):
                bboxes[i].x = res[i][2][0]
                bboxes[i].y = res[i][2][1]
                bboxes[i].width = res[i][2][2] - res[i][2][0]
                bboxes[i].height = res[i][2][3]- res[i][2][1]
                bboxes[i].example_id = example_id
                bboxes[i].label_id = 5
                bboxes[i].user_id = 1
                bboxes[i].save()
        else:
            bboxes = get_list_or_404(BoundingBox)
            res = run_detect(model_path, image_dir_path + file_name, 0.3, 0.5)
            for i in range(len(res)):
                BoundingBox.objects.create(
                    x = res[i][2][0],
                    y = res[i][2][1],
                    width = res[i][2][2] - res[i][2][0],
                    height = res[i][2][3]- res[i][2][1],
                    example_id = example_id,
                    label_id = 5,
                    user_id = 1,
                )


class BoundingBoxDetailAPI(BaseDetailAPI):
    queryset = BoundingBox.objects.all()
    serializer_class = BoundingBoxSerializer


class SegmentationListAPI(BaseListAPI):
    label_class = Segmentation
    serializer_class = SegmentationSerializer


class SegmentationDetailAPI(BaseDetailAPI):
    queryset = Segmentation.objects.all()
    serializer_class = SegmentationSerializer


    # queryset = DetectImage
def get_detect_info(request, *args, **kwargs):
    example_id = kwargs["example_id"]
    model_path = os.getcwd() + "/labels/auto_models/models/yanbao_paper30_CDLA-best.onnx"
    image_dir_path = os.getcwd() + "/media/"

    example = get_object_or_404(Example, id = example_id)
    file_name = str(example.filename)
    
    if BoundingBox.objects.filter(example_id = example_id):
        bboxes = get_list_or_404(BoundingBox, example_id = example_id)
        res = run_detect(model_path, image_dir_path + file_name, 0.3, 0.5)
        for i in range(len(res)):
            bboxes[i].x = res[i][2][0]
            bboxes[i].y = res[i][2][1]
            bboxes[i].width = res[i][2][2] - res[i][2][0]
            bboxes[i].height = res[i][2][3]- res[i][2][1]
            bboxes[i].example_id = example_id
            bboxes[i].label_id = 5
            bboxes[i].user_id = 1
            bboxes[i].save()
    else:
        bboxes = get_list_or_404(BoundingBox)
        res = run_detect(model_path, image_dir_path + file_name, 0.3, 0.5)
        for i in range(len(res)):
            BoundingBox.objects.create(
                x = res[i][2][0],
                y = res[i][2][1],
                width = res[i][2][2] - res[i][2][0],
                height = res[i][2][3]- res[i][2][1],
                example_id = example_id,
                label_id = 5,
                user_id = 1,
            )
    return HttpResponse(res)