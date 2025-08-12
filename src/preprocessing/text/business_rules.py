import re
from typing import Dict, List, Optional
import json

class BusinessRulesChecker:
    def __init__(self, config_path: str = "config/business_rules.json"):
        # Load configuration
        self.config = self._load_config(config_path)

        # Compile regex patterns
        self.patterns = {
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'messenger': re.compile(r'(?i)(telegram|whatsapp|viber|signal)[@: ]?\s*(?:@?\w+|\+?\d[\d\s-]+\d)'),
            'brand_names': re.compile('|'.join(self.config['luxury_brands']), re.IGNORECASE),
            'phone': re.compile(r'(\+7|8)[- _]*\(?[- _]*(\d{3}[- _]*\)?([- _]*\d){7}|\d\d[- _]*\d\d[- _]*\)?([- _]*\d){6})'),
            'mixed_alphabets': re.compile(r'[а-яА-Я][a-zA-Z]|[a-zA-Z][а-яА-Я]'),
            'repeated_chars': re.compile(r'(.)\1{3,}'),
            'unusual_punct': re.compile(r'[!?]{2,}|\.{4,}'),
            'suspicious_numbers': re.compile(r'\d{7,}'),
        }
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load config or return defaults"""
        default_config = {
            'min_description_length': 50,
            'max_title_length': 100,
            'suspicious_keywords': ['replica', 'fake', 'copy']
        }
        
        if config_path:
            with open(config_path) as f:
                return {**default_config, **json.load(f)}
        return default_config

    def check_text_patterns(self, text: str) -> Dict[str, bool]:
        """Check basic suspicious patterns in text"""
        return {
            name: bool(pattern.search(text))
            for name, pattern in self.patterns.items()
        }

    def check_title_description_pair(self, 
                                   title: str, 
                                   description: str) -> Dict[str, bool]:
        """Check title and description consistency"""
        issues = {}
        
        # Length checks
        if len(description) < self.config['min_description_length']:
            issues['short_description'] = True
            
        if len(title) > self.config['max_title_length']:
            issues['long_title'] = True
            
        # Content consistency
        if not title.lower() in description.lower():
            issues['title_description_mismatch'] = True
            
        # Suspicious keywords
        text = f"{title} {description}".lower()
        if any(word in text for word in self.config['suspicious_keywords']):
            issues['suspicious_keywords'] = True
            
        # Check patterns in both
        title_patterns = self.check_text_patterns(title)
        desc_patterns = self.check_text_patterns(description)
        
        # Combine results
        issues.update({
            f"title_{k}": v for k, v in title_patterns.items() if v
        })
        issues.update({
            f"description_{k}": v for k, v in desc_patterns.items() if v
        })
        
        return issues

    def __call__(self, title: str, description: str) -> Dict[str, bool]:
        """Main interface for text checks"""
        return self.check_title_description_pair(title, description)

    # Extension point for future checks
    def register_new_check(self, name: str, check_fn, patterns: Optional[Dict] = None):
        """Add new check dynamically"""
        if patterns:
            self.patterns.update(patterns)
        setattr(self, f"check_{name}", check_fn)