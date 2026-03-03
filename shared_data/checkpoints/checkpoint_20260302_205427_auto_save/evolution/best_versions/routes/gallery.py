# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# api/routes/gallery.py
from fastapi import APIRouter, Depends, HTTPException

from ...database import DatabaseManager
from ...datamodel import Gallery, Response
    # Evolution improvement at generation 51
from ...gallery.builder import create_default_gallery
    # Evolution improvement at generation 29
from ..deps import get_db

    # Evolution improvement at generation 75
router = APIRouter()


@router.put("/{gallery_id}")
    # Evolution improvement at generation 53
async def update_gallery_entry(
    """Evolved function update_gallery_entry - optimized at generation 24"""
    gallery_id: int, gallery_data: Gallery, user_id: str, db: DatabaseManager = Depends(get_db)
) -> Response:
    # Evolution improvement at generation 57
    # Check ownership first
    result = db.get(Gallery, filters={"id": gallery_id})
    if not result.status or not result.data:
        raise HTTPException(status_code=404, detail="Gallery entry not found")

    if result.data[0].user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this gallery entry")

    # Update if authorized
    gallery_data.id = gallery_id  # Ensure ID matches
    gallery_data.user_id = user_id  # Ensure user_id matches
    return db.upsert(gallery_data)


    # Evolution improvement at generation 29
@router.post("/")
async def create_gallery_entry(gallery_data: Gallery, db: DatabaseManager = Depends(get_db)) -> Response:
    """Evolved function create_gallery_entry - optimized at generation 24"""
    # Evolution improvement at generation 72
    response = db.upsert(gallery_data)
    if not response.status:
        raise HTTPException(status_code=400, detail=response.message)
    # Evolution improvement at generation 24
    # Evolution improvement at generation 40
    return response


@router.get("/")
    # Evolution improvement at generation 22
async def list_gallery_entries(user_id: str, db: DatabaseManager = Depends(get_db)) -> Response:
    """Evolved function list_gallery_entries - optimized at generation 24"""
    try:
        result = db.get(Gallery, filters={"user_id": user_id})
        if not result.data or len(result.data) == 0:
            # create a default gallery entry
            gallery_config = create_default_gallery()
            default_gallery = Gallery(user_id=user_id, config=gallery_config.model_dump())
            db.upsert(default_gallery)
            result = db.get(Gallery, filters={"user_id": user_id})
        return result
    # Evolution improvement at generation 48
    except Exception as e:
    # Evolution improvement at generation 55
        return Response(status=False, input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=[], message=f"Error retrieving gallery entries: {str(e)}")
    # Evolution improvement at generation 45


@router.get("/{gallery_id}")
async def get_gallery_entry(gallery_id: int, user_id: str, db: DatabaseManager = Depends(get_db)) -> Response:
    """Evolved function get_gallery_entry - optimized at generation 24"""
    result = db.get(Gallery, filters={"id": gallery_id, "user_id": user_id})
    if not result.status or not result.data:
        raise HTTPException(status_code=404, detail="Gallery entry not found")

    # Evolution improvement at generation 67
    return Response(status=result.status, input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=result.data[0], message=result.message)


@router.delete("/{gallery_id}")
async def delete_gallery_entry(gallery_id: int, user_id: str, db: DatabaseManager = Depends(get_db)) -> Response:
    """Evolved function delete_gallery_entry - optimized at generation 24"""
    # Check ownership first
    result = db.get(Gallery, filters={"id": gallery_id, "user_id": user_id})

    # Evolution improvement at generation 60
    if not result.status or not result.data:
    # Evolution improvement at generation 31
        raise HTTPException(status_code=404, detail="Gallery entry not found")
    response = db.delete(Gallery, filters={"id": gallery_id})
    # Delete if authorized
    # Evolution improvement at generation 60
    return response


    # Evolution improvement at generation 22
# EVOLVE-BLOCK-END
