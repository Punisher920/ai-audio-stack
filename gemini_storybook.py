"""Gemini Storybook Generator - Replicate official Storybook Gem

Generates illustrated 10-page storybooks using Gemini API.
Based on Google's Storybook Gem functionality.
"""

import os
import json
import google.generativeai as genai
from typing import List, Dict, Optional
import base64
from pathlib import Path


class GeminiStorybook:
    """Generate illustrated storybooks like the official Gemini Storybook Gem."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.0 Flash for fast, high-quality generation
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.image_model = genai.GenerativeModel('gemini-2.0-flash-exp')  # For image prompt generation
    
    def generate_story(self, prompt: str, pages: int = 10, style: str = "watercolor") -> Dict:
        """Generate complete storybook with text and image prompts.
        
        Args:
            prompt: Story description (e.g., "A dragon learning to fly")
            pages: Number of pages (default 10, like official Gem)
            style: Illustration style (watercolor, pixel art, comics, claymation, crochet, coloring book)
        
        Returns:
            Dictionary with title, pages (text + image prompts)
        """
        print(f"\n🎨 Generating {pages}-page storybook...")
        
        # Step 1: Generate story structure and text
        story_data = self._generate_story_text(prompt, pages)
        
        # Step 2: Generate image prompts for each page
        story_data = self._generate_image_prompts(story_data, style)
        
        print("✓ Storybook generated!")
        return story_data
    
    def _generate_story_text(self, prompt: str, pages: int) -> Dict:
        """Generate story title and text for each page."""
        
        system_instruction = f"""
You are a children's storybook writer. Create engaging, age-appropriate stories.

Generate a {pages}-page storybook based on the user's prompt.
Each page should have 2-3 sentences suitable for children.
Maintain consistent characters and plot progression.

Return ONLY valid JSON in this exact format:
{{
  "title": "Story Title",
  "pages": [
    {{"page_num": 1, "text": "Page 1 text here..."}},
    {{"page_num": 2, "text": "Page 2 text here..."}}
  ]
}}
"""
        
        full_prompt = f"{system_instruction}\n\nUser story prompt: {prompt}"
        
        response = self.model.generate_content(full_prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            story_data = json.loads(text)
            print(f"✓ Story text generated: {story_data['title']}")
            return story_data
        except Exception as e:
            print(f"Error parsing story: {e}")
            # Fallback structure
            return {
                "title": "A Magical Story",
                "pages": [{"page_num": i+1, "text": f"Page {i+1} of your story..."} for i in range(pages)]
            }
    
    def _generate_image_prompts(self, story_data: Dict, style: str) -> Dict:
        """Generate image prompts for each page based on story text."""
        
        print("✓ Generating image prompts...")
        
        style_instructions = {
            "watercolor": "soft watercolor painting, gentle brush strokes, pastel colors",
            "pixel art": "8-bit pixel art style, retro gaming aesthetic, vibrant colors",
            "comics": "comic book illustration, bold lines, dynamic action poses",
            "claymation": "claymation style, clay figures, tactile textures",
            "crochet": "crochet/yarn art style, soft textile textures",
            "coloring book": "black and white line art, coloring book style"
        }
        
        style_desc = style_instructions.get(style.lower(), "high-quality children's book illustration")
        
        for page in story_data["pages"]:
            # Generate image prompt for this page
            image_prompt_request = f"""
Based on this story page, create a detailed image generation prompt.

Page text: "{page['text']}"
Style: {style_desc}
Title: {story_data['title']}

Generate a single, concise image prompt (2-3 sentences) that:
- Captures the key scene/moment
- Maintains character consistency
- Specifies the artistic style
- Is suitable for children's book illustration

Return ONLY the image prompt text, no extra commentary.
"""
            
            response = self.image_model.generate_content(image_prompt_request)
            page["image_prompt"] = response.text.strip()
        
        print("✓ Image prompts generated")
        return story_data
    
    def save_storybook(self, story_data: Dict, output_path: str = "storybook.json"):
        """Save storybook data to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to {output_path}")
    
    def generate_with_images(self, story_data: Dict, image_generator_func) -> Dict:
        """Generate actual images using external image generation service.
        
        Args:
            story_data: Story with image prompts
            image_generator_func: Function that takes (prompt, style) and returns image data/path
        
        Returns:
            Updated story_data with image paths
        """
        print("\n🎨 Generating images...")
        
        for i, page in enumerate(story_data["pages"], 1):
            print(f"  Generating image {i}/{len(story_data['pages'])}...")
            image_result = image_generator_func(page["image_prompt"])
            page["image"] = image_result
        
        print("✓ All images generated!")
        return story_data


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    print("="*60)
    print("  GEMINI STORYBOOK GENERATOR")
    print("  Replicate official Storybook Gem functionality")
    print("="*60)
    
    # Initialize
    storybook = GeminiStorybook()
    
    # Example 1: Simple story generation
    story = storybook.generate_story(
        prompt="A friendly dragon learns to make friends at dragon school",
        pages=10,
        style="watercolor"
    )
    
    storybook.save_storybook(story, "dragon_story.json")
    
    # Print story preview
    print(f"\n📖 {story['title']}")
    print("=" * 40)
    for page in story['pages'][:3]:  # Show first 3 pages
        print(f"\nPage {page['page_num']}:")
        print(f"  Text: {page['text']}")
        print(f"  Image prompt: {page['image_prompt'][:80]}...")
    
    print("\n💡 Next steps:")
    print("1. Use image_prompts with Imagen, Stable Diffusion, or DALL-E")
    print("2. Integrate with your KDP pipeline")
    print("3. Add narration with OpenAI TTS or ElevenLabs")
    print("4. Generate QR codes for StorySpark!")
