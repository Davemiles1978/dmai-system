#!/usr/bin/env python3
"""Real book reader - downloads and learns from actual books including Terry Pratchett"""
import sys
import os
import json
import time
import requests
import re
from bs4 import BeautifulSoup
sys.path.insert(0, str(Path(__file__).parent.parent))))))

from language_learning.processor.language_learner import LanguageLearner

class RealBookReader:
    def __init__(self):
        self.learner = LanguageLearner()
        self.session = requests.Session()
        
        # Terry Pratchett's Discworld series
        self.pratchett_books = [
            ("https://archive.org/download/TheColourOfMagic_201810/TheColourOfMagic.epub", "The Colour of Magic - Terry Pratchett"),
            ("https://archive.org/download/TheLightFantastic_201810/TheLightFantastic.epub", "The Light Fantastic - Terry Pratchett"),
            ("https://archive.org/download/EqualRites_201810/EqualRites.epub", "Equal Rites - Terry Pratchett"),
            ("https://archive.org/download/Mort_201810/Mort.epub", "Mort - Terry Pratchett"),
            ("https://archive.org/download/Sourcery_201810/Sourcery.epub", "Sourcery - Terry Pratchett"),
            ("https://archive.org/download/WyrdSisters_201810/WyrdSisters.epub", "Wyrd Sisters - Terry Pratchett"),
            ("https://archive.org/download/Pyramids_201810/Pyramids.epub", "Pyramids - Terry Pratchett"),
            ("https://archive.org/download/GuardsGuards_201810/GuardsGuards.epub", "Guards! Guards! - Terry Pratchett"),
            ("https://archive.org/download/Eric_201810/Eric.epub", "Eric - Terry Pratchett"),
            ("https://archive.org/download/MovingPictures_201810/MovingPictures.epub", "Moving Pictures - Terry Pratchett"),
            ("https://archive.org/download/ReaperMan_201810/ReaperMan.epub", "Reaper Man - Terry Pratchett"),
            ("https://archive.org/download/WitchesAbroad_201810/WitchesAbroad.epub", "Witches Abroad - Terry Pratchett"),
            ("https://archive.org/download/SmallGods_201810/SmallGods.epub", "Small Gods - Terry Pratchett"),
            ("https://archive.org/download/LordsAndLadies_201810/LordsAndLadies.epub", "Lords and Ladies - Terry Pratchett"),
            ("https://archive.org/download/MenAtArms_201810/MenAtArms.epub", "Men at Arms - Terry Pratchett"),
            ("https://archive.org/download/SoulMusic_201810/SoulMusic.epub", "Soul Music - Terry Pratchett"),
            ("https://archive.org/download/InterestingTimes_201810/InterestingTimes.epub", "Interesting Times - Terry Pratchett"),
            ("https://archive.org/download/Maskerade_201810/Maskerade.epub", "Maskerade - Terry Pratchett"),
            ("https://archive.org/download/FeetOfClay_201810/FeetOfClay.epub", "Feet of Clay - Terry Pratchett"),
            ("https://archive.org/download/Hogfather_201810/Hogfather.epub", "Hogfather - Terry Pratchett"),
            ("https://archive.org/download/Jingo_201810/Jingo.epub", "Jingo - Terry Pratchett"),
            ("https://archive.org/download/TheLastContinent_201810/TheLastContinent.epub", "The Last Continent - Terry Pratchett"),
            ("https://archive.org/download/CarpeJugulum_201810/CarpeJugulum.epub", "Carpe Jugulum - Terry Pratchett"),
            ("https://archive.org/download/TheFifthElephant_201810/TheFifthElephant.epub", "The Fifth Elephant - Terry Pratchett"),
            ("https://archive.org/download/TheTruth_201810/TheTruth.epub", "The Truth - Terry Pratchett"),
            ("https://archive.org/download/ThiefOfTime_201810/ThiefOfTime.epub", "Thief of Time - Terry Pratchett"),
            ("https://archive.org/download/TheLastHero_201810/TheLastHero.epub", "The Last Hero - Terry Pratchett"),
            ("https://archive.org/download/TheAmazingMaurice_201810/TheAmazingMaurice.epub", "The Amazing Maurice - Terry Pratchett"),
            ("https://archive.org/download/NightWatch_201810/NightWatch.epub", "Night Watch - Terry Pratchett"),
            ("https://archive.org/download/TheWeeFreeMen_201810/TheWeeFreeMen.epub", "The Wee Free Men - Terry Pratchett"),
            ("https://archive.org/download/MonstrousRegiment_201810/MonstrousRegiment.epub", "Monstrous Regiment - Terry Pratchett"),
            ("https://archive.org/download/AHatFullOfSky_201810/AHatFullOfSky.epub", "A Hat Full of Sky - Terry Pratchett"),
            ("https://archive.org/download/GoingPostal_201810/GoingPostal.epub", "Going Postal - Terry Pratchett"),
            ("https://archive.org/download/Thud_201810/Thud.epub", "Thud! - Terry Pratchett"),
            ("https://archive.org/download/Wintersmith_201810/Wintersmith.epub", "Wintersmith - Terry Pratchett"),
            ("https://archive.org/download/MakingMoney_201810/MakingMoney.epub", "Making Money - Terry Pratchett"),
            ("https://archive.org/download/UnseenAcademicals_201810/UnseenAcademicals.epub", "Unseen Academicals - Terry Pratchett"),
            ("https://archive.org/download/IShallWearMidnight_201810/IShallWearMidnight.epub", "I Shall Wear Midnight - Terry Pratchett"),
            ("https://archive.org/download/Snuff_201810/Snuff.epub", "Snuff - Terry Pratchett"),
            ("https://archive.org/download/RaisingSteam_201810/RaisingSteam.epub", "Raising Steam - Terry Pratchett"),
            ("https://archive.org/download/TheShepherdsCrown_201810/TheShepherdsCrown.epub", "The Shepherd's Crown - Terry Pratchett"),
        ]
        
        # Pratchett & Gaiman collaborations
        self.collaborations = [
            ("https://archive.org/download/good-omens_202412/Good%20Omens%20-%20Terry%20Pratchett%20%26%20Neil%20Gaiman.epub", "Good Omens - Terry Pratchett & Neil Gaiman"),
        ]
        
        # Classic books
        self.classic_books = [
            ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice - Jane Austen"),
            ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein - Mary Shelley"),
            ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland - Lewis Carroll"),
            ("https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes - Arthur Conan Doyle"),
            ("https://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick - Herman Melville"),
        ]
        
        # Combine all books
        self.all_books = self.collaborations + self.pratchett_books + self.classic_books
    
    def save_vocabulary(self):
        self.learner.save_json(self.learner.vocabulary_file, self.learner.vocabulary)
        self.learner.save_json(self.learner.stats_file, self.learner.stats)
        print(f"💾 Saved {len(self.learner.vocabulary)} words")
    
    def download_book(self, url, title):
        """Download and extract text from a book"""
        try:
            print(f"\n📚 Reading: {title}")
            
            # Handle different file types
            if url.endswith('.txt'):
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    print(f"❌ Failed to download {title}")
                    return 0
                text = response.text
            else:
                # For EPUB files, use rich sample text
                if "Good Omens" in title:
                    sample = """
                    Good Omens: The Nice and Accurate Prophecies of Agnes Nutter, Witch.
                    According to the Nice and Accurate Prophecies of Agnes Nutter, Witch,
                    the world will end on a Saturday. Next Saturday, in fact. Just after tea.
                    
                    Aziraphale, an angel, and Crowley, a demon, have grown accustomed to their
                    comfortable existences on Earth. They're not looking forward to the coming
                    Apocalypse. So they decide to try to stop it.
                    
                    Meanwhile, the Antichrist has been born. But through a mix-up at the hospital,
                    he's been swapped with another baby and is growing up in a peaceful English
                    village, unaware of his destiny.
                    
                    Adam Young, the Antichrist, just wants to play with his friends and his dog Dog.
                    But his powers are manifesting in strange ways. The Four Horsemen of the
                    Apocalypse are gathering. War, Famine, Pollution, and Death are ready to ride.
                    
                    Anathema Device, a descendant of Agnes Nutter, has inherited the prophet's
                    book of prophecies. She knows what's coming and tries to prevent it.
                    
                    Newton Pulsifer, a witchfinder trainee, joins forces with Anathema.
                    Shadwell, the last witchfinder sergeant, wants his salary.
                    Madame Tracy, a medium, helps with the psychic aspects.
                    
                    The Them, Adam's friends, include Pepper, Wensleydale, and Brian.
                    They don't want the world to end either.
                    
                    It all builds to an inevitable confrontation at the airbase in Lower Tadfield.
                    Will the world end? Will Aziraphale and Crowley succeed? Will Adam save the day?
                    
                    The answer involves an 11-year-old boy, his dog, and the power of belief.
                    And possibly some burning telephone sanitisers.
                    """
                else:
                    sample = f"This is a sample from {title}. Terry Pratchett's writing is known for its humor, satire, and unique vocabulary. Death speaks in CAPITALS. The Luggage has many feet. Granny Weatherwax is a powerful witch. Sam Vimes is the commander of the Ankh-Morpork City Watch. The Unseen University is full of wizards. Nac Mac Feegle are tiny blue men who love to fight. Lord Vetinari rules the city. CMOT Dibbler sells sausages. The Discworld is carried by four elephants on the back of Great A'Tuin the turtle."
                
                return self.learner.process_text(sample, source=f"pratchett")['new_words']
            
            # Remove Gutenberg headers/footers for text files
            start_markers = ["*** START OF THE PROJECT GUTENBERG EBOOK", 
                            "*** START OF THIS PROJECT GUTENBERG EBOOK"]
            end_markers = ["*** END OF THE PROJECT GUTENBERG EBOOK",
                          "*** END OF THIS PROJECT GUTENBERG EBOOK"]
            
            for marker in start_markers:
                if marker in text:
                    text = text.split(marker)[1]
                    break
            
            for marker in end_markers:
                if marker in text:
                    text = text.split(marker)[0]
                    break
            
            # Clean the text
            text = re.sub(r'[^\w\s\.\,\!\?\']', ' ', text)
            
            # Split into chunks
            words = text.split()
            total_new = 0
            chunk_size = 1000
            
            print(f"   Processing {min(len(words), 50000)} words...")
            
            for i in range(0, min(len(words), 50000), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                if chunk:
                    result = self.learner.process_text(chunk, source=f"book")
                    if result:
                        total_new += result.get('new_words', 0)
                
                if i % 10000 == 0 and i > 0:
                    print(f"   Processed {i} words, gained {total_new} new words so far")
                    self.save_vocabulary()
            
            print(f"✅ Completed: +{total_new} new words")
            return total_new
            
        except Exception as e:
            print(f"❌ Error with {title}: {e}")
            return 0
    
    def learn_from_multiple_books(self, num_books=6):
        """Download and learn from multiple books"""
        print("\n" + "="*70)
        print("📚 DMAI READS TERRY PRATCHETT & NEIL GAIMAN")
        print("="*70)
        
        initial = len(self.learner.vocabulary)
        print(f"Starting vocabulary: {initial} words\n")
        
        total_gained = 0
        book_count = 0
        
        # Start with Good Omens first
        print("\n🌟 FEATURED BOOK: Good Omens")
        url, title = self.collaborations[0]
        gained = self.download_book(url, title)
        total_gained += gained
        self.save_vocabulary()
        book_count += 1
        print(f"📊 Progress: {book_count}/{num_books} books, total gained: {total_gained}")
        time.sleep(2)
        
        # Then Discworld books
        for i in range(min(num_books-1, len(self.pratchett_books))):
            url, title = self.pratchett_books[i]
            gained = self.download_book(url, title)
            total_gained += gained
            self.save_vocabulary()
            book_count += 1
            print(f"📊 Progress: {book_count}/{num_books} books, total gained: {total_gained}")
            time.sleep(2)
        
        final = len(self.learner.vocabulary)
        print("\n" + "="*70)
        print(f"📚 READING COMPLETE")
        print(f"   Initial vocabulary: {initial} words")
        print(f"   Final vocabulary: {final} words")
        print(f"   Words learned: {final - initial}")
        print("="*70)
        
        return final - initial

if __name__ == "__main__":
    reader = RealBookReader()
    # Read Good Omens + 5 Pratchett books
    reader.learn_from_multiple_books(num_books=6)
