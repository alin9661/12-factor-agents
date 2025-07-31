"""
Prompt management system.
Implements Factor 2: Own Your Prompts.

This system treats prompts as first-class code with templating,
version control, and testing capabilities.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class PromptTemplate(BaseModel):
    """A prompt template with metadata."""
    name: str
    version: str
    description: str
    template: str
    required_variables: List[str]
    created_at: datetime
    tags: List[str] = []


class PromptManager:
    """
    Manages prompts as first-class code.
    
    Features:
    - Template-based prompts with Jinja2
    - Version control for prompts
    - Variable validation
    - Prompt testing framework
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Path(__file__).parent / "templates"
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        self.logger = logger.bind(component="prompt_manager")
        self._template_cache: Dict[str, PromptTemplate] = {}

    def create_prompt_template(
        self,
        name: str,
        template: str,
        description: str = "",
        version: str = "1.0.0",
        tags: List[str] = None,
    ) -> PromptTemplate:
        """Create a new prompt template."""
        
        # Extract required variables from template
        required_variables = self._extract_template_variables(template)
        
        prompt_template = PromptTemplate(
            name=name,
            version=version,
            description=description,
            template=template,
            required_variables=required_variables,
            created_at=datetime.utcnow(),
            tags=tags or [],
        )
        
        # Save template to file
        self._save_template(prompt_template)
        
        # Cache the template
        self._template_cache[name] = prompt_template
        
        self.logger.info(
            "Created prompt template",
            name=name,
            version=version,
            variables=required_variables,
        )
        
        return prompt_template

    def load_template(self, name: str) -> PromptTemplate:
        """Load a prompt template by name."""
        
        # Check cache first
        if name in self._template_cache:
            return self._template_cache[name]
        
        # Load from file
        template_path = self.prompts_dir / f"{name}.jinja"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template '{name}' not found")
        
        # Load template content and metadata
        template_content = template_path.read_text()
        metadata_path = self.prompts_dir / f"{name}.json"
        
        if metadata_path.exists():
            import json
            metadata = json.loads(metadata_path.read_text())
            prompt_template = PromptTemplate(
                template=template_content,
                **metadata
            )
        else:
            # Create minimal template if no metadata exists
            prompt_template = PromptTemplate(
                name=name,
                version="1.0.0",
                description="",
                template=template_content,
                required_variables=self._extract_template_variables(template_content),
                created_at=datetime.utcnow(),
            )
        
        # Cache and return
        self._template_cache[name] = prompt_template
        return prompt_template

    def render_prompt(
        self,
        template_name: str,
        variables: Dict[str, Any],
        validate_variables: bool = True,
    ) -> str:
        """Render a prompt template with variables."""
        
        template = self.load_template(template_name)
        
        if validate_variables:
            self._validate_variables(template, variables)
        
        try:
            jinja_template = self.jinja_env.from_string(template.template)
            rendered = jinja_template.render(**variables)
            
            self.logger.debug(
                "Rendered prompt",
                template=template_name,
                variables=list(variables.keys()),
                length=len(rendered),
            )
            
            return rendered
            
        except Exception as e:
            self.logger.error(
                "Failed to render prompt",
                template=template_name,
                error=str(e),
                exc_info=True,
            )
            raise

    def list_templates(self) -> List[str]:
        """List all available prompt templates."""
        template_files = list(self.prompts_dir.glob("*.jinja"))
        return [f.stem for f in template_files]

    def test_template(
        self,
        template_name: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Test a prompt template with various inputs."""
        
        template = self.load_template(template_name)
        results = {
            "template": template_name,
            "test_results": [],
            "passed": 0,
            "failed": 0,
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                rendered = self.render_prompt(template_name, test_case, validate_variables=False)
                results["test_results"].append({
                    "case": i + 1,
                    "status": "passed",
                    "input": test_case,
                    "output_length": len(rendered),
                    "output_preview": rendered[:200] + "..." if len(rendered) > 200 else rendered,
                })
                results["passed"] += 1
                
            except Exception as e:
                results["test_results"].append({
                    "case": i + 1,
                    "status": "failed",
                    "input": test_case,
                    "error": str(e),
                })
                results["failed"] += 1
        
        self.logger.info(
            "Template test completed",
            template=template_name,
            passed=results["passed"],
            failed=results["failed"],
        )
        
        return results

    def _extract_template_variables(self, template: str) -> List[str]:
        """Extract variable names from a Jinja2 template."""
        try:
            parsed = self.jinja_env.parse(template)
            variables = set()
            
            for node in parsed.find_all():
                if hasattr(node, 'name') and isinstance(node.name, str):
                    variables.add(node.name)
            
            # Filter out Jinja2 built-ins and functions
            builtin_names = {
                'range', 'lipsum', 'dict', 'cycler', 'joiner', 'namespace'
            }
            variables = variables - builtin_names
            
            return sorted(list(variables))
            
        except Exception as e:
            logger.warning("Failed to extract template variables", error=str(e))
            return []

    def _validate_variables(self, template: PromptTemplate, variables: Dict[str, Any]) -> None:
        """Validate that all required variables are provided."""
        missing_vars = set(template.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(
                f"Missing required variables for template '{template.name}': {missing_vars}"
            )

    def _save_template(self, template: PromptTemplate) -> None:
        """Save a template to disk."""
        
        # Save template content
        template_path = self.prompts_dir / f"{template.name}.jinja"
        template_path.write_text(template.template)
        
        # Save metadata
        metadata_path = self.prompts_dir / f"{template.name}.json"
        metadata = template.model_dump(exclude={"template"})
        metadata["created_at"] = metadata["created_at"].isoformat()
        
        import json
        metadata_path.write_text(json.dumps(metadata, indent=2))


# Global prompt manager instance
prompt_manager = PromptManager()