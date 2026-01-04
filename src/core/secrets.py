from decouple import config


class SingletonClass(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance


class SecretManager(SingletonClass):
    def get_from_env(self, key: str, default=None, cast=str):
        """Gets an environment variable. If the variable does not exist, it returns the default value instead.
        Args:
            key (str): The key of the environment variable.
            default (any, optional): The value to return if the variable does not exist. Defaults to None.
            cast (function, optional): The type to cast the value to. Defaults to str.
        Returns:
            any: The value of the environment variable. If the variable does not exist, the default value is returned.
        """
        value = config(key, default=default, cast=cast)
        return value if value is not None else default
    

