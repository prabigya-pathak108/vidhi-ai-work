import os
import shutil
from abc import ABC, abstractmethod

class StorageProvider(ABC):
    @abstractmethod
    def save(self, content: bytes, filename: str) -> str: pass
    @abstractmethod
    def exists(self, filename: str) -> bool: pass

class LocalStorage(StorageProvider):
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, content: bytes, filename: str) -> str:
        full_path = os.path.join(self.base_path, filename)
        with open(full_path, "wb") as f:
            f.write(content)
        return full_path

    def exists(self, filename: str) -> bool:
        return os.path.exists(os.path.join(self.base_path, filename))

class StorageFactory:
    @staticmethod
    def get_storage(provider_type: str, **kwargs):
        if provider_type == "local":
            return LocalStorage(kwargs.get("path", "data/raw"))
        # Add S3Storage here later
        raise ValueError(f"Unsupported storage: {provider_type}")