"""add llm_credentials table and avatars.llm_credential_id

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-17
"""
from alembic import op
import sqlalchemy as sa

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'llm_credentials',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('user_id', sa.String(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        sa.Column('api_key', sa.String(), nullable=False),
        sa.Column('api_base_url', sa.String(), nullable=True),
        sa.Column('kindroid_ai_id', sa.String(), nullable=True),
        sa.Column('is_favorite', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('sort_order', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_verify_ok', sa.Boolean(), nullable=True),
        sa.Column('last_verify_message', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_llm_credentials_user_id', 'llm_credentials', ['user_id'])
    op.create_index('ix_llm_credentials_user_sort', 'llm_credentials', ['user_id', 'sort_order'])

    op.add_column('avatars', sa.Column('llm_credential_id', sa.String(), nullable=True))
    op.create_foreign_key(
        'fk_avatars_llm_credential_id', 'avatars', 'llm_credentials',
        ['llm_credential_id'], ['id'], ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint('fk_avatars_llm_credential_id', 'avatars', type_='foreignkey')
    op.drop_column('avatars', 'llm_credential_id')
    op.drop_index('ix_llm_credentials_user_sort', table_name='llm_credentials')
    op.drop_index('ix_llm_credentials_user_id', table_name='llm_credentials')
    op.drop_table('llm_credentials')
