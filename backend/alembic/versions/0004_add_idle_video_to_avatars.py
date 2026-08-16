"""add idle_video_url to avatars

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-14
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('avatars', sa.Column('idle_video_url', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('avatars', 'idle_video_url')
