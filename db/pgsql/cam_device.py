from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from core.config import settings


engine = create_async_engine(
    settings.postgres_dsn,
    pool_size=20,
    max_overflow=0,
    pool_recycle=30,
    pool_pre_ping=True,
    pool_use_lifo=True,
)


SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)
