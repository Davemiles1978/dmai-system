#!/usr/bin/env python3
"""
Alex Riviera - Complete Publishing System
Includes: How-to-Draw, Paint by Numbers, Coloring Books, Children's Books, Instructional, Novels
All books can be sold as digital downloads or print-on-demand
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, '/Users/davidmiles/Desktop/dmai-system')

from components.alex_riviera.book_generator import BookGenerator
from components.alex_riviera.publishing_orchestrator import AlexRivieraPublishing

def main():
    print("="*70)
    print("📚 ALEX RIVIERA - COMPLETE PUBLISHING SYSTEM")
    print("="*70)
    print("\n👤 Alex Riviera, 28, Writer & Producer")
    print("📧 alex.riviera.creator@proton.me")
    print("\n⚠️ All projects require your approval before sending")
    print("💰 All books can be sold as digital downloads (PDF) or print-on-demand")
    print("="*70)
    
    book_gen = BookGenerator()
    publisher = AlexRivieraPublishing()
    
    print("\n" + "="*70)
    print("📖 GENERATING ARTSY BOOKS (High-margin digital products)")
    print("="*70)
    
    # Generate Artsy Books (can be sold repeatedly)
    artsy_books = [
        ('how_to_draw', "Cats"),
        ('how_to_draw', "Dragons"),
        ('how_to_draw', None),  # Random
        ('paint_by_numbers', "Landscapes"),
        ('paint_by_numbers', "Animals"),
        ('paint_by_numbers', None),
        ('coloring_book', "Mandala Patterns"),
        ('coloring_book', "Magical Unicorns"),
        ('coloring_book', None)
    ]
    
    for book_type, param in artsy_books:
        print(f"\n🎨 Generating {book_type.replace('_', ' ').upper()} book...")
        
        if book_type == 'how_to_draw':
            if param:
                book = book_gen.generate_how_to_draw_book(param)
            else:
                book = book_gen.generate_how_to_draw_book()
        elif book_type == 'paint_by_numbers':
            if param:
                book = book_gen.generate_paint_by_numbers_book(param)
            else:
                book = book_gen.generate_paint_by_numbers_book()
        else:
            if param:
                book = book_gen.generate_coloring_book(param)
            else:
                book = book_gen.generate_coloring_book()
        
        approval = publisher.submit_for_approval(book, 'book')
        print(f"   ✅ '{book['title']}' - Ready for review")
        print(f"   💰 {book.get('commercial_notes', 'High-margin product')[:80]}...")
    
    # Generate Traditional Books
    print("\n" + "="*70)
    print("📖 GENERATING TRADITIONAL BOOKS")
    print("="*70)
    
    traditional_books = [
        ('childrens', None),
        ('instructional', None),
        ('novel', 'fiction'),
        ('novel', 'sci_fi')
    ]
    
    for book_type, param in traditional_books:
        print(f"\n📝 Generating {book_type.upper()} book...")
        
        if book_type == 'childrens':
            book = book_gen.generate_childrens_book()
        elif book_type == 'instructional':
            book = book_gen.generate_instructional_book()
        else:
            book = book_gen.generate_novel(param)
        
        approval = publisher.submit_for_approval(book, 'book')
        print(f"   ✅ '{book['title']}' - Ready for review")
    
    # Show summary
    print("\n" + "="*70)
    print("📋 SUMMARY - AWAITING YOUR APPROVAL")
    print("="*70)
    
    pending = publisher.get_pending_approvals()
    print(f"\n📚 Books pending review: {len(pending)}")
    
    for approval in pending:
        project = approval['project_data']
        book_type = project.get('type', 'unknown')
        commercial_note = project.get('commercial_notes', '')
        
        print(f"\n   📖 {project.get('title', 'Untitled')}")
        print(f"      Type: {book_type.upper()}")
        print(f"      Pages: {project.get('pages', 'N/A')}")
        print(f"      ID: {approval['id']}")
        if commercial_note and '✓' in commercial_note:
            print(f"      💰 {commercial_note[:100]}...")
        print(f"      Files: data/alex_projects/for_review/{approval['id']}/")
    
    print("\n" + "="*70)
    print("✅ TO APPROVE A PROJECT:")
    print("   python3 approve_project.py <project_id> approve")
    print("\n❌ TO REJECT WITH NOTES:")
    print("   python3 approve_project.py <project_id> reject \"Your notes here\"")
    print("\n💰 ARTSY BOOKS COMMERCIAL VALUE:")
    print("   • How-to-Draw: $5-15 per digital download")
    print("   • Paint by Numbers: $4-10 per digital download")  
    print("   • Coloring Books: $3-8 per digital download")
    print("   • Unlimited reproductions - sell repeatedly!")
    print("="*70)
    
    # Save summary
    import json
    from datetime import datetime
    
    summary = {
        'generated_books': [{'title': b['project_data']['title'], 'type': b['project_data'].get('type')} for b in pending],
        'pending_approvals': [{'id': a['id'], 'title': a['title']} for a in pending],
        'commercial_potential': {
            'how_to_draw': 'High - Evergreen content, multiple subjects',
            'paint_by_numbers': 'Very High - Relaxation market growing',
            'coloring_books': 'Highest - Adult coloring market booming',
            'traditional_books': 'Variable - Depends on genre and marketing'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/alex_projects/generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n📄 Summary saved to: data/alex_projects/generation_summary.json")

if __name__ == "__main__":
    main()
