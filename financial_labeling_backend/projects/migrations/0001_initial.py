# Generated by Django 4.1.10 on 2024-02-24 13:29

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("roles", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("contenttypes", "0002_remove_content_type_name"),
    ]

    operations = [
        migrations.CreateModel(
            name="Project",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("description", models.TextField(default="")),
                ("guideline", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "project_type",
                    models.CharField(
                        choices=[
                            ("DocumentClassification", "Document Classification"),
                            ("SequenceLabeling", "Sequence Labeling"),
                            ("Seq2seq", "Seq2Seq"),
                            (
                                "IntentDetectionAndSlotFilling",
                                "Intent Detection And Slot Filling",
                            ),
                            ("Speech2text", "Speech2Text"),
                            ("ImageClassification", "Image Classification"),
                            ("BoundingBox", "Bounding Box"),
                            ("Segmentation", "Segmentation"),
                            ("ImageCaptioning", "Image Captioning"),
                        ],
                        max_length=30,
                    ),
                ),
                ("random_order", models.BooleanField(default=False)),
                ("collaborative_annotation", models.BooleanField(default=False)),
                ("single_class_classification", models.BooleanField(default=False)),
                (
                    "allow_member_to_create_label_type",
                    models.BooleanField(default=False),
                ),
                (
                    "created_by",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "polymorphic_ctype",
                    models.ForeignKey(
                        editable=False,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="polymorphic_%(app_label)s.%(class)s_set+",
                        to="contenttypes.contenttype",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
        ),
        migrations.CreateModel(
            name="BoundingBoxProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="ImageCaptioningProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="ImageClassificationProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="IntentDetectionAndSlotFillingProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="SegmentationProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="Seq2seqProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="SequenceLabelingProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
                ("allow_overlapping", models.BooleanField(default=False)),
                ("grapheme_mode", models.BooleanField(default=False)),
                ("use_relation", models.BooleanField(default=False)),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="Speech2textProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="TextClassificationProject",
            fields=[
                (
                    "project_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="projects.project",
                    ),
                ),
            ],
            options={
                "abstract": False,
                "base_manager_name": "objects",
            },
            bases=("projects.project",),
        ),
        migrations.CreateModel(
            name="Tag",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("text", models.TextField()),
                (
                    "project",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="tags",
                        to="projects.project",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Member",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "project",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="role_mappings",
                        to="projects.project",
                    ),
                ),
                (
                    "role",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="roles.role"
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="role_mappings",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "unique_together": {("user", "project")},
            },
        ),
    ]
