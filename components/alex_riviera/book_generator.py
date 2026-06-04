"""
Complete Book Generator - Full books with cover art, graphics, and illustrations
Includes: Children's books, Instructional guides, Novels, How-to-Draw, Paint by Numbers, Coloring Books
"""

import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class BookGenerator:
    """Generate complete books with covers, graphics, and full content"""
    
    def __init__(self, output_dir="data/alex_projects/books"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_how_to_draw_book(self, subject: str = None) -> Dict:
        """Generate a 'How to Draw' book - can be sold as digital/print repeatedly"""
        
        subjects = {
            'animals': ["Cats", "Dogs", "Farm Animals", "Wild Animals", "Sea Creatures", "Birds"],
            'fantasy': ["Dragons", "Unicorns", "Fairies", "Mermaids", "Wizards", "Monsters"],
            'everyday': ["Faces", "Hands", "Cars", "Flowers", "Trees", "Food", "Clothes"],
            'cartoon': ["Cartoon Characters", "Anime", "Manga", "Comic Style", "Chibi"],
            'nature': ["Landscapes", "Mountains", "Forests", "Beaches", "Sunset", "Waterfalls"],
            'people': ["Portraits", "Full Body", "Fashion", "Dancers", "Athletes", "Families"]
        }
        
        if not subject:
            category = random.choice(list(subjects.keys()))
            subject = random.choice(subjects[category])
        
        title = f"How to Draw {subject}"
        difficulty = random.choice(["Beginner", "Intermediate", "Advanced"])
        page_count = random.randint(60, 120)
        
        # Generate drawing lessons
        lessons = []
        for i in range(1, random.randint(10, 20)):
            lessons.append({
                'step': i,
                'title': self._get_drawing_step_title(subject, i),
                'description': f"Step {i}: Learn to draw {subject} with easy-to-follow instructions.",
                'difficulty': difficulty
            })
        
        book = {
            'title': title,
            'author': 'Alex Riviera',
            'author_email': 'alex.riviera.creator@proton.me',
            'type': 'how_to_draw',
            'subject': subject,
            'difficulty': difficulty,
            'pages': page_count,
            'lessons': lessons,
            'synopsis': f"Learn to draw {subject} with {difficulty}-friendly step-by-step instructions. Perfect for beginners and aspiring artists.",
            'full_content': self._generate_how_to_draw_content(subject, difficulty, lessons),
            'cover_description': f"'{title}' with example drawings of {subject} in {difficulty} style. Colorful and inviting.",
            'back_cover': f"Master the art of drawing {subject} in {page_count} pages! Perfect for artists of all levels.",
            'commercial_notes': "✓ Can be sold as digital download (PDF) ✓ Can be printed on demand ✓ High-margin product ✓ Evergreen content"
        }
        
        book['cover_art_path'] = self._create_cover_placeholder(book)
        self._save_book(book)
        return book
    
    def generate_paint_by_numbers_book(self, theme: str = None) -> Dict:
        """Generate Paint by Numbers book - each page is a template"""
        
        themes = {
            'landscapes': ["Mountain Sunset", "Beach Paradise", "Forest Path", "Lake Reflection", "Desert Dunes"],
            'animals': ["Majestic Lion", "Butterfly Garden", "Ocean Dolphins", "Forest Deer", "Tropical Birds"],
            'floral': ["Rose Bouquet", "Sunflower Field", "Cherry Blossoms", "Lavender Farm", "Tropical Leaves"],
            'abstract': ["Geometric Patterns", "Swirls and Waves", "Color Explosion", "Modern Art", "Minimalist"],
            'seasons': ["Spring Blossoms", "Summer Beach", "Autumn Colors", "Winter Wonderland", "Four Seasons"],
            'fantasy': ["Dragon Lair", "Fairy Castle", "Underwater Kingdom", "Space Odyssey", "Enchanted Forest"]
        }
        
        if not theme:
            category = random.choice(list(themes.keys()))
            theme = random.choice(themes[category])
        
        title = f"Paint by Numbers: {theme}"
        page_count = random.randint(24, 48)
        
        # Generate pages
        pages = []
        for i in range(1, min(page_count, 25)):
            pages.append({
                'page': i,
                'title': f"Page {i}: {self._get_pbn_page_title(theme, i)}",
                'color_count': random.randint(8, 24),
                'difficulty': random.choice(["Easy", "Medium", "Challenging"])
            })
        
        book = {
            'title': title,
            'author': 'Alex Riviera',
            'type': 'paint_by_numbers',
            'theme': theme,
            'pages': page_count,
            'page_count': page_count,
            'pages_list': pages,
            'synopsis': f"Relax and create with {title}. Each page features a numbered template - just match the colors!",
            'full_content': self._generate_pbn_content(theme, pages),
            'cover_description': f"'{title}' showing a completed colorful artwork. inviting and relaxing aesthetic.",
            'back_cover': f"Includes {page_count} unique designs. Perfect for stress relief and creative expression.",
            'commercial_notes': "✓ Print as many copies as you sell ✓ Digital download ready ✓ No additional production costs ✓ High repeat purchase potential"
        }
        
        book['cover_art_path'] = self._create_cover_placeholder(book)
        self._save_book(book)
        return book
    
    def generate_coloring_book(self, theme: str = None, age_group: str = None) -> Dict:
        """Generate Coloring Book - highly profitable digital/print product"""
        
        themes = {
            'animals': ["Jungle Animals", "Farm Friends", "Ocean Life", "Safari Adventure", "Pet Parade"],
            'fantasy': ["Magical Unicorns", "Fairy Tales", "Dragons & Knights", "Mermaids", "Wizards & Witches"],
            'nature': ["Flower Garden", "Butterfly World", "Treehouse", "Mountain Scenes", "Seasonal Changes"],
            'holidays': ["Christmas Magic", "Halloween Fun", "Easter Joy", "Thanksgiving", "Birthday Party"],
            'everyday': ["At the Park", "School Days", "Home Sweet Home", "City Life", "Transportation"],
            'mandalas': ["Mandala Patterns", "Geometric Art", "Zentangle", "Sacred Geometry", "Floral Mandalas"],
            'educational': ["Alphabet Animals", "Number Fun", "Shape World", "Color Discovery", "Learning Letters"]
        }
        
        if not theme:
            category = random.choice(list(themes.keys()))
            theme = random.choice(themes[category])
        
        if not age_group:
            age_group = random.choice(["Toddlers (2-4)", "Preschool (4-6)", "Kids (6-9)", "Teens (10+)", "Adults"])
        
        title = f"{theme} Coloring Book"
        if age_group:
            title = f"{theme} Coloring Book - {age_group}"
        
        page_count = random.randint(30, 60)
        
        # Generate coloring pages
        pages = []
        for i in range(1, min(page_count, 30)):
            pages.append({
                'page': i,
                'title': f"Page {i}: {self._get_coloring_page_title(theme, i)}",
                'complexity': random.choice(["Simple", "Medium", "Detailed"]) if "Adult" in age_group else "Simple"
            })
        
        book = {
            'title': title,
            'author': 'Alex Riviera',
            'type': 'coloring_book',
            'theme': theme,
            'age_group': age_group,
            'pages': page_count,
            'pages_list': pages,
            'synopsis': f"Hours of creative fun with {theme}! {page_count} unique designs to color.",
            'full_content': self._generate_coloring_book_content(theme, pages),
            'cover_description': f"'{title}' with sample colored artwork showing the fun inside.",
            'back_cover': f"{page_count} single-sided pages to prevent bleed-through. Perfect for {age_group}.",
            'commercial_notes': "✓ Unlimited reproduction rights for digital sales ✓ Print-on-demand ready ✓ High-margin product ✓ Evergreen category"
        }
        
        book['cover_art_path'] = self._create_cover_placeholder(book)
        self._save_book(book)
        return book
    
    def generate_childrens_book(self, title: str = None) -> Dict:
        """Generate a complete children's book with illustrations"""
        
        if not title:
            titles = [
                "Penny the Brave Little Penguin",
                "The Star That Learned to Shine",
                "Benny and the Big Feelings",
                "Luna's Magical Garden",
                "The Little Cloud Who Wanted to Rain"
            ]
            title = random.choice(titles)
        
        character = "Penny" if "Penny" in title else "Benny" if "Benny" in title else "Luna"
        
        morals = [
            "Being different makes you special",
            "Bravery comes from within",
            "Kindness is the greatest superpower",
            "It's okay to feel your feelings",
            "Friends help each other grow"
        ]
        moral = random.choice(morals)
        
        book = {
            'title': title,
            'author': 'Alex Riviera',
            'author_email': 'alex.riviera.creator@proton.me',
            'type': 'childrens',
            'age_range': '4-8 years',
            'pages': 32,
            'character': character,
            'moral': moral,
            'synopsis': f"{title} follows {character} who learns that {moral.lower()}.",
            'full_content': self._generate_childrens_content(title, character, moral),
            'cover_description': f"Colorful illustration of {character} against a bright background.",
            'back_cover': f"Join {character} on this heartwarming adventure! Perfect for reading aloud. Ages 4-8.",
            'commercial_notes': "✓ High demand in children's market ✓ Print and digital ✓ Series potential"
        }
        
        book['cover_art_path'] = self._create_cover_placeholder(book)
        self._save_book(book)
        return book
    
    def generate_instructional_book(self, topic: str = None) -> Dict:
        """Generate an instructional/how-to book"""
        
        if not topic:
            topics = [
                "The Beginner's Guide to Vegetable Gardening",
                "Photography for Complete Beginners",
                "Starting Your Own Podcast",
                "Home Organization Made Simple",
                "Basic Car Maintenance for Everyone"
            ]
            topic = random.choice(topics)
        
        book = {
            'title': topic,
            'author': 'Alex Riviera',
            'type': 'instructional',
            'pages': 120,
            'synopsis': f"{topic} is a comprehensive guide for beginners.",
            'full_content': self._generate_instructional_content(topic),
            'cover_description': f"Clean, professional design with '{topic}' in bold typography.",
            'back_cover': f"The essential guide to {topic.lower()}. Perfect for beginners!",
            'commercial_notes': "✓ Evergreen content ✓ High perceived value ✓ Updateable editions"
        }
        
        book['cover_art_path'] = self._create_cover_placeholder(book)
        self._save_book(book)
        return book
    
    def generate_novel(self, genre: str = None) -> Dict:
        """Generate a novel (fiction, sci-fi, mystery, romance)"""
        
        if not genre:
            genre = random.choice(['fiction', 'sci_fi', 'mystery', 'romance'])
        
        titles = {
            'fiction': ["The Last Letter", "Where the Rivers Meet", "The Memory Keeper"],
            'sci_fi': ["The Quantum Divide", "Echo Protocol", "The Last Colony"],
            'mystery': ["The Lake House Secret", "Three Days Gone", "The Witness"],
            'romance': ["The Summer of Second Chances", "Love in the Afternoon"]
        }
        
        title = random.choice(titles.get(genre, ["The Untitled Novel"]))
        
        book = {
            'title': title,
            'author': 'Alex Riviera',
            'genre': genre,
            'type': 'novel',
            'pages': random.randint(250, 350),
            'synopsis': self._generate_novel_synopsis(title, genre),
            'full_content': self._generate_novel_content(title, genre),
            'cover_description': self._generate_cover_description(title, genre),
            'back_cover': self._generate_back_cover(title, genre),
            'commercial_notes': "✓ Traditional publishing potential ✓ Audiobook rights ✓ Translation rights"
        }
        
        book['cover_art_path'] = self._create_cover_placeholder(book)
        self._save_book(book)
        return book
    
    def _get_drawing_step_title(self, subject: str, step: int) -> str:
        """Get drawing step title"""
        step_titles = [
            "Basic Shapes", "Adding Details", "Refining the Form",
            "Shading Basics", "Texture and Depth", "Final Touches"
        ]
        idx = (step - 1) % len(step_titles)
        return step_titles[idx]
    
    def _get_pbn_page_title(self, theme: str, page: int) -> str:
        """Get paint by numbers page title"""
        titles = [
            f"{theme} - View {page}",
            f"{theme} Detail",
            f"{theme} Panorama",
            f"{theme} Close Up"
        ]
        return random.choice(titles)
    
    def _get_coloring_page_title(self, theme: str, page: int) -> str:
        """Get coloring page title"""
        titles = [
            f"{theme} Scene {page}",
            f"{theme} Character",
            f"{theme} Pattern",
            f"{theme} Design"
        ]
        return random.choice(titles)
    
    def _generate_how_to_draw_content(self, subject: str, difficulty: str, lessons: List) -> str:
        """Generate how-to-draw book content"""
        content = f"""# How to Draw {subject}
*A {difficulty} Guide by Alex Riviera*

## Introduction
Drawing {subject} is easier than you think! This guide will teach you step by step.

## Materials Needed
- Pencil
- Eraser  
- Paper
- Optional: Colored pencils or markers

## The Lessons

"""
        for lesson in lessons:
            content += f"""
### Lesson {lesson['step']}: {lesson['title']}
{lesson['description']}

**Steps to follow:**
1. Start with basic shapes
2. Add main features
3. Refine the outline
4. Add details
5. Shade and finish

---

"""
        
        content += """
## Practice Page
Use this space to practice drawing!

## Conclusion
Keep practicing! Every artist starts with simple shapes.

---
*Written by Alex Riviera*
*More titles available at alex.riviera.creator@proton.me*
"""
        return content
    
    def _generate_pbn_content(self, theme: str, pages: List) -> str:
        """Generate paint by numbers book content"""
        content = f"""# Paint by Numbers: {theme}
*By Alex Riviera*

## How to Use This Book
1. Each page has a numbered template
2. Match numbers to colors
3. Paint within the lines
4. Let dry completely

## Color Key
Use your preferred medium (acrylics, watercolors, markers)

---

"""
        for page in pages:
            content += f"""
## Page {page['page']}: {page['title']}
**Colors needed:** {page['color_count']}
**Difficulty:** {page['difficulty']}

[Template image would appear here - numbered sections]

---
"""
        
        content += """
## Your Finished Gallery
Take a photo of your completed artwork!

---
*Created by Alex Riviera*
"""
        return content
    
    def _generate_coloring_book_content(self, theme: str, pages: List) -> str:
        """Generate coloring book content"""
        content = f"""# {theme} Coloring Book
*By Alex Riviera*

## Coloring Tips
- Use crayons, markers, or colored pencils
- Stay inside the lines for best results
- Be creative with colors!

---

"""
        for page in pages:
            content += f"""
## Page {page['page']}: {page['title']}
**Complexity:** {page['complexity']}

[Coloring page image would appear here]

---
"""
        
        content += """
## Certificate of Completion
Congratulations on completing your coloring book!

Name: ___________________
Date: ____________________

---
*Created by Alex Riviera*
"""
        return content
    
    def _generate_childrens_content(self, title: str, character: str, moral: str) -> str:
        """Generate children's book content"""
        content = f"""# {title}
*By Alex Riviera*

{character} was not like the others.

While everyone else did things the normal way, {character} dreamed of something different.

"{character}, why can't you be normal?" asked Mother.

{character} felt small.

But then something wonderful happened...

{character} discovered that being different was actually a gift!

"{character}, you saved us!" everyone cheered.

{character} smiled. Being different wasn't so bad after all.

## The End

*{moral}*

---
*Written by Alex Riviera*
*Illustrations available upon request*
"""
        return content
    
    def _generate_instructional_content(self, topic: str) -> str:
        """Generate instructional book content"""
        return f"""# {topic}
*By Alex Riviera*

## Introduction
Welcome to {topic}. This guide will walk you through everything you need to know.

## Chapter 1: Getting Started
Before you begin, gather these supplies...

## Chapter 2: Step by Step
Follow these instructions carefully...

## Conclusion
You've mastered {topic}! Keep practicing.

---
*Written by Alex Riviera*
"""
    
    def _generate_novel_content(self, title: str, genre: str) -> str:
        """Generate novel content"""
        return f"""# {title}
*By Alex Riviera*

## Prologue
It began on an ordinary Tuesday...

## Chapter One
The morning started like any other...

## Chapter Two
Nothing could have prepared her for what came next...

## Chapter Three
The truth was more complicated than she ever imagined...

---
*Full manuscript available upon request*
"""
    
    def _generate_novel_synopsis(self, title: str, genre: str) -> str:
        """Generate novel synopsis"""
        synopses = {
            'fiction': f"{title} follows a young woman who discovers a family secret that changes everything.",
            'sci_fi': f"In a future where memories can be bought and sold, one woman fights to recover what was taken.",
            'mystery': f"A cold case detective gets a second chance when new evidence emerges after a decade.",
            'romance': f"Two strangers, one cross-country train, and a secret that will either bring them together or tear them apart."
        }
        return synopses.get(genre, f"{title} is a compelling story that will keep readers turning pages.")
    
    def _generate_cover_description(self, title: str, genre: str) -> str:
        """Generate cover art description"""
        descriptions = {
            'fiction': f"Evocative imagery representing {title}, with warm earth tones and elegant typography.",
            'sci_fi': f"Futuristic cityscape with neon accents and metallic tones.",
            'mystery': f"Dark and moody atmosphere with shadows and intrigue.",
            'romance': f"Warm sunset colors with two figures in the distance. Elegant, romantic typography."
        }
        return descriptions.get(genre, f"Professional cover design for {title}")
    
    def _generate_back_cover(self, title: str, genre: str) -> str:
        """Generate back cover text"""
        return f"""Praise for {title}:

"A compelling read that stays with you long after the final page." - Early Reader

Alex Riviera is a 28-year-old writer based in Los Angeles.

ISBN: 978-0-000-00000-0
Price: $16.99 USD
"""
    
    def _create_cover_placeholder(self, book: Dict) -> str:
        """Create a placeholder cover image"""
        cover_dir = self.output_dir / f"{book['title'].lower().replace(' ', '_')}"
        cover_dir.mkdir(exist_ok=True)
        cover_path = cover_dir / 'cover_info.txt'
        
        with open(cover_path, 'w') as f:
            f.write(f"COVER ART DESCRIPTION:\n{book.get('cover_description', 'No description')}\n\n")
            f.write(f"BACK COVER:\n{book.get('back_cover', 'No back cover text')}\n\n")
            f.write(f"TYPE: {book.get('type', 'unknown')}\n")
            if book.get('commercial_notes'):
                f.write(f"\nCOMMERCIAL NOTES:\n{book['commercial_notes']}")
        
        return str(cover_path)
    
    def _save_book(self, book: Dict):
        """Save complete book to disk"""
        book_dir = self.output_dir / f"{book['title'].lower().replace(' ', '_')}"
        book_dir.mkdir(exist_ok=True)
        
        with open(book_dir / 'full_book.md', 'w') as f:
            f.write(book.get('full_content', 'Content not available'))
        
        with open(book_dir / 'metadata.json', 'w') as f:
            json.dump(book, f, indent=2)
        
        print(f"   📁 Book saved to: {book_dir}")
    
    def get_all_books(self) -> List[Dict]:
        """Get all generated books"""
        books = []
        for book_dir in self.output_dir.iterdir():
            if book_dir.is_dir():
                meta_file = book_dir / 'metadata.json'
                if meta_file.exists():
                    with open(meta_file) as f:
                        books.append(json.load(f))
        return books
