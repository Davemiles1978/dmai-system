# How to Generate Alex Riviera's Final Avatar

## Using Midjourney (Best Quality)
1. Go to Discord and join a Midjourney server
2. Use this prompt:
/imagine Professional photograph of Alex Riviera, a 28-year-old Valhalla girl with Viking styling. Mousy blonde hair (soft muted blonde), shoulder-length, slightly wavy. Very green almond-shaped eyes. Tanned skin with scattered freckles (girl-next-door look). Athletic muscular build with D cup bust. Strong Viking bone structure. Wearing a casual fitted sweater. Confident, approachable expression. Natural lighting, neutral background. High resolution, 4K, photorealistic. --ar 4:5 --style raw --v 6

text

## Using DALL-E 3
1. Go to https://chat.openai.com or https://labs.openai.com
2. Use the DALL-E 3 prompt from avatar_prompts.json
3. Generate multiple variants and select the best match to your reference images

## Using Stable Diffusion (Local)
1. If you have Stable Diffusion installed locally:
2. Use the prompt from avatar_prompts.json
3. Use the negative prompt provided
4. Generate at 1024x1280 resolution
5. Use a photorealistic model (Realistic Vision, Juggernaut XL)

## How to Provide Feedback
Once you generate an image you're happy with:

1. Save it as `data/avatars/final_avatar.png`
2. Run the consistency locking script:
```bash
python3 -c "
from components.media.AvatarConsistencyEngine import AvatarConsistencyEngine
engine = AvatarConsistencyEngine()
# After generating, we'll lock the specs
print('Ready to lock avatar once final image is saved')
"
What to Look For
✅ Hair is mousy blonde (soft, muted - not bright platinum)

✅ Eyes are VERY green (striking)

✅ Freckles visible on tanned skin

✅ Athletic build visible

✅ D cup bust proportion (as in reference image "Girl on the left's breasts.png")

✅ Viking bone structure in face

✅ Girl-next-door expression (approachable, warm)

Once Confirmed
The final avatar will be LOCKED and used for ALL future content across ALL platforms.
