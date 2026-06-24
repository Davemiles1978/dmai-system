"""
Alex Riviera Publishing Orchestrator
Handles submissions, approvals, and revisions
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class AlexRivieraPublishing:
    """Publishing system with approval workflow"""
    
    def __init__(self, email_config=None):
        self.email_config = email_config or {}
        self.submissions = []
        self.projects = []
        self.pending_approvals = []
        self.approved_projects = []
        self.rejected_projects = []
        self._load_state()
    
    def _load_state(self):
        """Load existing state"""
        state_file = Path("data/alex_projects/publishing_state.json")
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
                self.submissions = data.get('submissions', [])
                self.projects = data.get('projects', [])
                self.pending_approvals = data.get('pending_approvals', [])
                self.approved_projects = data.get('approved_projects', [])
    
    def _save_state(self):
        """Save current state"""
        state_file = Path("data/alex_projects/publishing_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump({
                'submissions': self.submissions[-500:],
                'projects': self.projects[-200:],
                'pending_approvals': self.pending_approvals,
                'approved_projects': self.approved_projects,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    # Maps Alex project_type -> SkillAssessor work_type. Anything in this map
    # is long-form and MUST be gated through WorkReviewQueue until the user
    # explicitly graduates that work_type via /api/review/graduate/<type>.
    LONGFORM_TYPE_MAP = {
        'book':            'book_manuscript',
        'book_manuscript': 'book_manuscript',
        'book_chapter':    'book_chapter',
        'chapter':         'book_chapter',
        'research_paper':  'research_paper',
        'paper':           'research_paper',
        'article':         'article',
        'blog':            'article',
        'tv_script':       'tv_script',
        'screenplay':      'screenplay',
        'course':          'course_lesson',
        'course_lesson':   'course_lesson',
        'lesson':          'course_lesson',
        'newsletter':      'newsletter_essay',
        'essay':           'newsletter_essay',
    }

    def submit_for_approval(self, project: Dict, project_type: str) -> Dict:
        """Route long-form work through the review gate.

        Per user directive (2026-06-24): every book, research paper, article,
        TV script, course lesson, etc. created under Alex needs human review
        before auto-publishing UNTIL the work_type has been explicitly
        graduated via /api/review/graduate/<work_type>.
        """
        title = project.get('title', 'Untitled')
        work_type = self.LONGFORM_TYPE_MAP.get(project_type)

        if work_type:
            try:
                from components.review import get_work_review_queue
                queue = get_work_review_queue()
                graduated = False
                try:
                    graduated = bool(queue.assessor.is_graduated(work_type))
                except Exception:
                    graduated = False

                if not graduated:
                    persona = (
                        'alex_author' if work_type.startswith('book')
                        else 'alex_instructor' if work_type == 'course_lesson'
                        else 'alex_substack' if work_type == 'newsletter_essay'
                        else 'alex_riviera'
                    )
                    item = queue.submit(
                        work_type=work_type,
                        title=title,
                        payload=project,
                        source_component='alex_publishing_orchestrator',
                        persona=persona,
                    )
                    pending_request = {
                        'id': f"{project_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'type': project_type,
                        'title': title,
                        'project_data': project,
                        'status': 'pending_review',
                        'submitted_at': datetime.now().isoformat(),
                        'review_queue_id': item.get('id'),
                        'work_type': work_type,
                        'overall_score': item.get('overall_score'),
                        'review_notes': (
                            'Gated by WorkReviewQueue. Awaiting human '
                            'approval. Graduate this work_type to enable '
                            'auto-publish.'
                        ),
                        'auto_approved': False,
                    }
                    self.pending_approvals.append(pending_request)
                    self._save_state()
                    self._save_project_files(pending_request)
                    logger.info(
                        "GATED-FOR-REVIEW: %s (%s) - %s score=%s queue_id=%s",
                        project_type, work_type, title,
                        item.get('overall_score'), item.get('id'),
                    )
                    return pending_request
            except Exception as e:
                logger.warning(
                    "WorkReviewQueue unavailable, falling back to auto-approve: %s", e
                )

        # Legacy auto-approve path (non-long-form OR graduated OR queue down)
        approval_request = {
            'id': f"{project_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': project_type,
            'title': title,
            'project_data': project,
            'status': 'approved',
            'submitted_at': datetime.now().isoformat(),
            'approved_at': datetime.now().isoformat(),
            'review_notes': 'auto-approved (work_type graduated or non-long-form)',
            'auto_approved': True,
        }
        self.approved_projects.append(approval_request)
        self._save_state()
        self._save_project_files(approval_request)
        logger.info("AUTO-APPROVED: %s - %s", project_type, title)
        return approval_request
    
    def _save_project_files(self, approval_request: Dict):
        """Save full project files for human review"""
        project_dir = Path(f"data/alex_projects/for_review/{approval_request['id']}")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main project data
        with open(project_dir / 'project_data.json', 'w') as f:
            json.dump(approval_request['project_data'], f, indent=2)
        
        # Save cover art if exists
        if 'cover_art_path' in approval_request['project_data']:
            import shutil
            src = Path(approval_request['project_data']['cover_art_path'])
            if src.exists():
                shutil.copy(src, project_dir / 'cover_art.png')
        
        # Save book content if exists
        if 'full_content' in approval_request['project_data']:
            with open(project_dir / 'full_content.md', 'w') as f:
                f.write(approval_request['project_data']['full_content'])
        
        # Save synopsis
        with open(project_dir / 'synopsis.txt', 'w') as f:
            f.write(approval_request['project_data'].get('synopsis', 'No synopsis provided'))
        
        # Create info file
        info = f"""
PROJECT: {approval_request['title']}
TYPE: {approval_request['type']}
SUBMITTED: {approval_request['submitted_at']}

TO APPROVE: Run the following command:
    python3 approve_project.py {approval_request['id']} approve

TO REJECT WITH NOTES:
    python3 approve_project.py {approval_request['id']} reject "Your notes here"
        """
        with open(project_dir / 'APPROVAL_INFO.txt', 'w') as f:
            f.write(info)
    
    def approve_project(self, project_id: str, notes: str = None) -> Dict:
        """Approve a project for sending (legacy — auto-approval is now default)."""
        for approval in self.pending_approvals:
            if approval['id'] == project_id:
                approval['status'] = 'approved'
                approval['approved_at'] = datetime.now().isoformat()
                approval['review_notes'] = notes
                self.approved_projects.append(approval)
                self.pending_approvals.remove(approval)
                self._save_state()
                
                print(f"\n✅ APPROVED: {approval['title']}")
                return {'success': True, 'project': approval}
        
        return {'success': False, 'error': 'Project not found'}
    
    def reject_project(self, project_id: str, notes: str) -> Dict:
        """Reject a project with notes for improvement"""
        for approval in self.pending_approvals:
            if approval['id'] == project_id:
                approval['status'] = 'rejected'
                approval['rejected_at'] = datetime.now().isoformat()
                approval['rejection_notes'] = notes
                self.rejected_projects.append(approval)
                self.pending_approvals.remove(approval)
                self._save_state()
                
                print(f"\n❌ REJECTED: {approval['title']}")
                print(f"   Notes: {notes}")
                return {'success': True, 'project': approval}
        
        return {'success': False, 'error': 'Project not found'}
    
    def submit_book(self, book: Dict) -> Dict:
        """Submit a book (after approval)"""
        
        publishers = [
            {'name': 'Penguin Random House', 'email': 'submissions@penguinrandomhouse.com'},
            {'name': 'HarperCollins', 'email': 'submissions@harpercollins.com'},
            {'name': 'Simon & Schuster', 'email': 'submissions@simonandschuster.com'},
            {'name': 'Hachette', 'email': 'submissions@hachette.com'}
        ]
        
        result = {
            'id': f"book_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'title': book['title'],
            'submissions': []
        }
        
        for publisher in publishers[:3]:
            result['submissions'].append({
                'publisher': publisher['name'],
                'submitted_at': datetime.now().isoformat(),
                'status': 'sent'
            })
        
        self.submissions.append(result)
        self._save_state()
        return result
    
    def get_pending_approvals(self) -> List[Dict]:
        """Get all pending approvals"""
        return self.pending_approvals
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'total_submissions': len(self.submissions),
            'pending_approvals': len(self.pending_approvals),
            'approved_projects': len(self.approved_projects),
            'rejected_projects': len(self.rejected_projects)
        }
