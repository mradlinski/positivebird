"""empty message

Revision ID: 2b9b26c5d71
Revises: 2d340813392
Create Date: 2015-11-18 22:55:18.275232

"""

# revision identifiers, used by Alembic.
revision = '2b9b26c5d71'
down_revision = '2d340813392'

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('data_label', 'ip')
    op.drop_column('user_lookup', 'ip')
    op.drop_column('visit', 'ip')
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.add_column('visit', sa.Column('ip', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('user_lookup', sa.Column('ip', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('data_label', sa.Column('ip', sa.TEXT(), autoincrement=False, nullable=True))
    ### end Alembic commands ###
