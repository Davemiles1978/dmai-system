#!/usr/bin/env python3
"""
Alex Riviera - Autonomous Publishing System
Generates diverse content, checks for plagiarism, submits to publishers
All communications appear from Alex Riviera, a human writer
"""

import sys
import time
import random
from datetime import datetime

sys.path.insert(0, '/Users/davidmiles/Desktop/dmai-system')

from components.alex_riviera.content_generator import AlexRivieraContent
from components.alex_riviera.publishing_orchestrator import AlexRivieraPublishing
from components.plagiarism.ContentValidator import ContentValidator

def main():
    print("="*70)
    print("📚 ALEX RIVIERA - AUTONOMOUS PUBLISHING SYSTEM")
    print("="*70)
    print("\n👤 Identity: Alex Riviera, 28, Writer & Producer")
    print("📧 Email: alex.riviera.creator@proton.me")
    print("📍 Location: Los Angeles, CA")
    print("\n⚠️ All content is checked for plagiarism before submission")
    print("⚠️ No AI mentions - Alex Riviera is presented as human creator")
    print("="*70)
    
    # Initialize
    content_gen = AlexRivieraContent()
    publisher = AlexRivieraPublishing()
    validator = ContentValidator()
    
    # Generate diverse content
    print("\n" + "="*70)
    print("📖 GENERATING DIVERSE CONTENT")
    print("="*70)
    
    # List of genres to generate
    book_genres = ['fiction', 'sci_fi', 'mystery', 'romance', 'comedy', 'horror', 'childrens', 'non_fiction']
    tv_genres = ['drama', 'thriller', 'sitcom', 'sci_fi', 'mystery', 'horror']
    
    books_generated = []
    tv_series_generated = []
    
    # Generate books in different genres
    print("\n📚 GENERATING BOOKS...")
    for genre in book_genres[:5]:  # Generate 5 books
        print(f"\n   Writing {genre.upper()} book...")
        book, validation = content_gen.generate_and_validate_book(genre)
        
        if validation['is_valid']:
            print(f"   ✅ '{book['title']}' - VALIDATED (no issues)")
            books_generated.append(book)
        else:
            print(f"   ⚠️ '{book['title']}' - NEEDS REVISION")
            for issue in validation.get('issues', []):
                print(f"      - {issue}")
        
        time.sleep(1)  # Rate limiting
    
    # Generate TV series in different genres
    print("\n\n🎬 GENERATING TV SERIES...")
    for genre in tv_genres[:4]:  # Generate 4 TV series
        print(f"\n   Writing {genre.upper()} TV series...")
        series, validation = content_gen.generate_and_validate_tv_series(genre)
        
        if validation['is_valid']:
            print(f"   ✅ '{series['title']}' - VALIDATED (no issues)")
            tv_series_generated.append(series)
        else:
            print(f"   ⚠️ '{series['title']}' - NEEDS REVISION")
            for issue in validation.get('issues', []):
                print(f"      - {issue}")
        
        time.sleep(1)
    
    # Show verification summary
    print("\n" + "="*70)
    print("✅ VERIFICATION SUMMARY")
    print("="*70)
    
    summary = content_gen.get_verification_summary()
    print(f"\n   Total works generated: {summary['total_generated']}")
    print(f"   Verified & ready to submit: {summary['verified_ready']}")
    print(f"   Needs revision: {summary['needs_revision']}")
    
    if summary['verified_ready'] > 0:
        print("\n   Verified works:")
        for title in summary['verified_list']:
            print(f"      - {title}")
    
    # Submit verified books
    print("\n" + "="*70)
    print("📤 SUBMITTING VERIFIED BOOKS")
    print("="*70)
    
    for book in books_generated:
        print(f"\n   Submitting '{book['title']}'...")
        result = publisher.submit_book(book)
        print(f"   ✅ Submitted to {len(result['submissions'])} publishers")
        for sub in result['submissions']:
            print(f"      - {sub['publisher']}")
    
    # Submit verified TV series
    print("\n" + "="*70)
    print("📺 SUBMITTING VERIFIED TV SERIES")
    print("="*70)
    
    for series in tv_series_generated:
        print(f"\n   Submitting '{series['title']}'...")
        result = publisher.submit_tv_series(series)
        print(f"   ✅ Submitted to {len(result['submissions'])} studios")
        for sub in result['submissions']:
            print(f"      - {sub['studio']}")
    
    # Final status
    print("\n" + "="*70)
    print("📊 PUBLISHING STATUS")
    print("="*70)
    
    status = publisher.get_status()
    print(f"\n   Total submissions: {status['total_submissions']}")
    print(f"   Pending responses: {status['pending_responses']}")
    print(f"   Active projects: {status['projects']}")
    
    print("\n" + "="*70)
    print("✅ ALEX RIVIERA PUBLISHING SYSTEM ACTIVE")
    print("")
    print("📁 All emails logged to: data/alex_outreach/sent_emails.log")
    print("📁 Verified works saved to: data/validation/verified_works.json")
    print("")
    print("⚠️ To send real emails, set:")
    print("   export GMAIL_USER='alex.riviera.creator@proton.me'")
    print("   export GMAIL_APP_PASSWORD='your-proton-app-password'")
    print("="*70)
    
    # Save summary report
    import json
    report = {
        'creator': 'Alex Riviera',
        'email': 'alex.riviera.creator@proton.me',
        'generated_books': [{'title': b['title'], 'genre': b['genre']} for b in books_generated],
        'generated_tv': [{'title': t['title'], 'genre': t['genre']} for t in tv_series_generated],
        'verification_summary': summary,
        'publishing_status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/alex_publishing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n📄 Report saved: data/alex_publishing_report.json")

if __name__ == "__main__":
    main()
