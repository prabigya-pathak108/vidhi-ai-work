import os
from pathlib import Path
from typing import Dict

class PromptFactory:
    _templates: Dict[str, str] = {}
    # Base path relative to this file: ../prompts/
    _base_path = Path(__file__).parent.parent / "prompts"

    @classmethod
    def get_prompt(cls, template_name: str, **kwargs) -> str:
        """
        Reads a template and formats it. 
        Caches the template in memory for performance.
        """
        if template_name not in cls._templates:
            file_path = cls._base_path / f"{template_name}.tmpl"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Template {template_name}.tmpl not found at {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                cls._templates[template_name] = f.read()

        template = cls._templates[template_name]
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Prompt template '{template_name}' requires variable: {e}")