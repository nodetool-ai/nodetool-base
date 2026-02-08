# nodetool.nodes.gemini.audio

## TextToSpeech

Generate speech audio from text using Google's Gemini text-to-speech models.

This node converts text input into natural-sounding speech audio using Google's
advanced text-to-speech models with support for multiple voices and speech styles.

Supported voices:
- achernar, achird, algenib, algieba, alnilam
- aoede, autonoe, callirrhoe, charon, despina
- enceladus, erinome, fenrir, gacrux, iapetus
- kore, laomedeia, leda, orus, puck
- pulcherrima, rasalgethi, sadachbia, sadaltager, schedar
- sulafat, umbriel, vindemiatrix, zephyr, zubenelgenubi

Use cases:
- Create voiceovers for videos and presentations
- Generate audio content for podcasts and audiobooks
- Add voice narration to applications
- Create accessibility features with speech output
- Generate multilingual audio content

**Tags:** google, text-to-speech, tts, audio, speech, voice, ai

**Fields:**
- **text**: The text to convert to speech. (str)
- **model**: The text-to-speech model to use (TTSModel)
- **voice_name**: The voice to use for speech generation (VoiceName)
- **style_prompt**: Optional style prompt to control speech characteristics (e.g., 'Say cheerfully', 'Speak with excitement') (str)


## ImageToVideo

Generate videos from images using Google's Veo models.

This node uses Google's Veo models to animate static images into dynamic videos.
Supports 720p resolution at 24fps with 8-second duration and native audio generation.

Use cases:
- Animate still artwork and photographs
- Create dynamic social media content from images
- Generate product showcase videos from photos
- Transform static graphics into engaging animations
- Create video presentations from slide images

**Tags:** google, video, generation, image-to-video, veo, ai, animation

**Fields:**
- **image**: The image to animate into a video (ImageRef)
- **prompt**: Optional text prompt describing the desired animation (str)
- **model**: The Veo model to use for video generation (VeoModel)
- **aspect_ratio**: The aspect ratio of the generated video (VeoAspectRatio)
- **negative_prompt**: Negative prompt to guide what to avoid in the video (str)


## TextToVideo

Generate videos from text prompts using Google's Veo models.

This node uses Google's Veo models to generate high-quality videos from text descriptions.
Supports 720p resolution at 24fps with 8-second duration and native audio generation.

Use cases:
- Create cinematic clips from text descriptions
- Generate social media video content
- Produce marketing and promotional videos
- Visualize creative concepts and storyboards
- Create animated content with accompanying audio

**Tags:** google, video, generation, text-to-video, veo, ai

**Fields:**
- **prompt**: The text prompt describing the video to generate (str)
- **model**: The Veo model to use for video generation (VeoModel)
- **aspect_ratio**: The aspect ratio of the generated video (VeoAspectRatio)
- **negative_prompt**: Negative prompt to guide what to avoid in the video (str)

## nodetool.nodes.nodetool.model3d

## TextTo3D

Generate 3D assets from text prompts using supported 3D providers.

Use cases:
- Create 3D mockups from text descriptions
- Bootstrap 3D assets for AR/VR workflows

**Tags:** 3d, generation, ai, text-to-3d

**Fields:**
- **provider**: Provider identifier (str)
- **model**: Provider model id (str)
- **prompt**: Text prompt describing the 3D asset (str)
- **guidance_scale**: Classifier-free guidance scale (float)
- **num_inference_steps**: Number of denoising steps (int)
- **seed**: Random seed (-1 for random) (int)

## ImageTo3D

Generate 3D assets from images using supported 3D providers.

Use cases:
- Reconstruct 3D assets from reference images
- Convert product photos into 3D models

**Tags:** 3d, generation, ai, image-to-3d

**Fields:**
- **provider**: Provider identifier (str)
- **model**: Provider model id (str)
- **image**: Input image to reconstruct (ImageRef)
- **prompt**: Optional prompt to guide generation (str)
- **guidance_scale**: Classifier-free guidance scale (float)
- **num_inference_steps**: Number of denoising steps (int)
- **seed**: Random seed (-1 for random) (int)


## ImageGeneration

Generate an image using Google's Imagen model via the Gemini API.

Use cases:
- Create images from text descriptions
- Generate assets for creative projects
- Explore AI-powered image synthesis

**Tags:** google, image generation, ai, imagen

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **model**: The image generation model to use (ImageGenerationModel)
- **image**: The image to use as a base for the generation. (ImageRef)


## GroundedSearch

Search the web using Google's Gemini API with grounding capabilities.

This node uses Google's Gemini API to perform web searches and return structured results
with source information. Requires a Gemini API key.

Use cases:
- Research current events and latest information
- Find reliable sources for fact-checking
- Gather web-based information with citations
- Get up-to-date information beyond the model's training data

**Tags:** google, search, grounded, web, gemini, ai

**Fields:**
- **query**: The search query to execute (str)
- **model**: The Gemini model to use for search (GeminiModel)


## Suno

Generate music using Suno AI via Kie.ai.
kie, suno, music, audio, ai, generation, vocals, instrumental

Creates full tracks with vocals and instrumentals up to around 8 minutes long.
Supports the latest Suno V4.5+ model with improved vocal quality and composition.

Use cases:
- Generate background music for projects
- Create AI-composed songs with vocals
- Produce instrumentals for content
- Generate music in various genres and styles

**Tags:** 

**Fields:**
- **prompt**: Description of the music to generate (genre, mood, instruments, etc.). (str)
- **lyrics**: Optional lyrics for the song. Leave empty for instrumental. (str)
- **style**: Music style/genre. Use 'custom' for prompt-based generation. (Style)
- **instrumental**: Generate instrumental-only (no vocals). (bool)
- **duration**: Approximate duration in seconds. (int)
- **model**: Suno model version to use. (Model)


## GrokImagineImageToVideo

Generate videos from images using xAI's Grok Imagine model via Kie.ai.
kie, grok, xai, video generation, ai, image-to-video, multimodal

Grok Imagine can transform images into videos with coherent motion and
synchronized background elements.

Use cases:
- Animate static images
- Create videos from photos
- Generate motion from images

**Tags:** 

**Fields:**
- **image**: The source image to animate. (ImageRef)
- **prompt**: Optional text to guide the animation. (str)
- **duration**: Duration of the generated video. (Duration)


## GrokImagineTextToVideo

Generate videos from text using xAI's Grok Imagine model via Kie.ai.
kie, grok, xai, video generation, ai, text-to-video, multimodal

Grok Imagine can generate videos from text prompts with coherent motion
and synchronized background elements.

Use cases:
- Generate videos from text descriptions
- Create cinematic content from prompts
- Text-to-video generation

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the video to generate. (str)
- **duration**: Duration of the generated video. (Duration)
- **resolution**: Video resolution. (Resolution)


## HailuoImageToVideoPro

Generate videos from images using MiniMax's Hailuo 2.3 Pro model via Kie.ai.
kie, hailuo, minimax, video generation, ai, image-to-video, pro, high-quality

Hailuo 2.3 Pro specializes in realistic character motion, expressive facial
micro-expressions, and cinematic quality for image-to-video generation.

Use cases:
- Generate high-quality cinematic videos from images
- Create content with expressive facial animations
- Professional image-to-video production
- Complex character movements from static images

**Tags:** 

**Fields:**
- **image**: The reference image to animate into a video. (ImageRef)
- **prompt**: Optional text to guide the video generation. (str)
- **resolution**: Video resolution. (Resolution)


## HailuoImageToVideoStandard

Generate videos from images using MiniMax's Hailuo 2.3 Standard model via Kie.ai.
kie, hailuo, minimax, video generation, ai, image-to-video, standard, fast

Hailuo 2.3 Standard offers efficient image-to-video generation with good quality
and faster processing times for practical use cases.

Use cases:
- Generate quality videos from images efficiently
- Quick image animation with realistic motion
- Fast image-to-video prototyping
- Practical video content creation

**Tags:** 

**Fields:**
- **image**: The reference image to animate into a video. (ImageRef)
- **prompt**: Optional text to guide the video generation. (str)
- **resolution**: Video resolution. (Resolution)


## KlingAIAvatarPro

Generate talking avatar videos using Kuaishou's Kling AI via Kie.ai.
kie, kling, kuaishou, avatar, video generation, ai, talking-head, lip-sync

Transforms a photo plus audio track into a lip-synced talking avatar video
with natural-looking speech animation and consistent identity.

Use cases:
- Create virtual influencer content
- Generate educational presenters
- Lip-synced avatar videos
- Virtual spokesperson creation

**Tags:** 

**Fields:**
- **image**: The face/character image to animate. (ImageRef)
- **audio**: The audio track for lip-syncing. (AudioRef)
- **prompt**: Optional text to guide emotions and expressions. (str)
- **mode**: Generation mode: 'standard' or 'pro' for higher quality. (Mode)


## KlingAIAvatarStandard

Generate talking avatar videos using Kuaishou's Kling AI via Kie.ai.
kie, kling, kuaishou, avatar, video generation, ai, talking-head, lip-sync

Transforms a photo plus audio track into a lip-synced talking avatar video
with natural-looking speech animation and consistent identity.

Use cases:
- Create virtual influencer content
- Generate educational presenters
- Lip-synced avatar videos
- Virtual spokesperson creation

**Tags:** 

**Fields:**
- **image**: The face/character image to animate. (ImageRef)
- **audio**: The audio track for lip-syncing. (AudioRef)
- **prompt**: Optional text to guide emotions and expressions. (str)
- **mode**: Generation mode: 'standard' or 'pro' for higher quality. (Mode)


## KlingImageToVideo

Generate videos from images using Kuaishou's Kling 2.6 model via Kie.ai.
kie, kling, kuaishou, video generation, ai, image-to-video, realistic

Kling 2.6 Image-to-Video transforms reference images into high-quality videos
with realistic motion, physics consistency, and cinematic quality output.

Use cases:
- Generate high-quality videos from images
- Create realistic motion and physics from images
- Animate static images with cinematic quality
- Professional image-to-video production

**Tags:** 

**Fields:**
- **image**: The reference image to animate into a video. (ImageRef)
- **prompt**: Optional text to guide the video generation. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **resolution**: Video resolution. (Resolution)
- **duration**: Video duration in seconds. (int)
- **seed**: Random seed for reproducible results. Use -1 for random seed. (int)
- **sound**: Whether the generated video includes sound. (bool)


## KlingTextToVideo

Generate videos from text using Kuaishou's Kling 2.6 model via Kie.ai.
kie, kling, kuaishou, video generation, ai, text-to-video, realistic

Kling 2.6 offers high-quality text-to-video generation with realistic motion,
physics consistency, and cinematic quality output.

Use cases:
- Generate high-quality videos from text
- Create realistic motion and physics
- Cinematic content creation
- Professional video production

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the video to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **resolution**: Video resolution. (Resolution)
- **duration**: Video duration in seconds. (int)
- **seed**: Random seed for reproducible results. Use -1 for random seed. (int)


## SeedanceV1LiteImageToVideo

Generate videos from images using ByteDance's Seedance V1 Lite model via Kie.ai.
kie, seedance, bytedance, video generation, ai, image-to-video, lite, fast

Seedance V1 Lite offers fast image-to-video generation with efficient processing,
transforming reference images into dynamic video sequences.

Use cases:
- Fast image animation from reference images
- Quick video prototyping
- Animate static images efficiently
- Rapid iteration on image-to-video concepts

**Tags:** 

**Fields:**
- **image**: The reference image to animate into a video. (ImageRef)
- **prompt**: Optional text to guide the video generation. (str)
- **resolution**: Video resolution. (Resolution)
- **duration**: Video duration in seconds. (int)
- **camera_fixed**: Whether to keep the camera fixed or allow camera movement. (bool)
- **seed**: Random seed for reproducible results. Use -1 for random seed. (int)
- **enable_safety_checker**: Enable safety checker to filter inappropriate content. (bool)


## SeedanceV1LiteTextToVideo

Generate videos from text using ByteDance's Seedance V1 Lite model via Kie.ai.
kie, seedance, bytedance, video generation, ai, text-to-video, lite, fast

Seedance V1 Lite offers fast text-to-video generation with efficient processing,
supporting multi-shot videos and cinematic scene transitions.

Use cases:
- Fast video prototyping from text
- Create multi-shot videos with scene transitions
- Quick concept visualization
- Rapid iteration on video ideas

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the video to generate. Supports shot descriptions with [Cut to] notation. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **resolution**: Video resolution. (Resolution)
- **duration**: Video duration in seconds. (int)
- **camera_fixed**: Whether to keep the camera fixed or allow camera movement. (bool)
- **seed**: Random seed for reproducible results. Use -1 for random seed. (int)
- **enable_safety_checker**: Enable safety checker to filter inappropriate content. (bool)


## SeedanceV1ProFastImageToVideo

Generate videos from images using ByteDance's Seedance V1 Pro Fast model via Kie.ai.
kie, seedance, bytedance, video generation, ai, image-to-video, pro, fast

Seedance V1 Pro Fast offers balanced image-to-video generation with both
quality and speed, transforming reference images into high-quality video sequences
with optimized processing time.

Use cases:
- Generate high-quality videos from images efficiently
- Fast professional image animations
- Balanced quality and speed image-to-video conversion
- Production-ready image animation

**Tags:** 

**Fields:**
- **image**: The reference image to animate into a video. (ImageRef)
- **prompt**: Optional text to guide the video generation. (str)
- **resolution**: Video resolution. (Resolution)
- **duration**: Video duration in seconds. (int)
- **camera_fixed**: Whether to keep the camera fixed or allow camera movement. (bool)
- **seed**: Random seed for reproducible results. Use -1 for random seed. (int)
- **enable_safety_checker**: Enable safety checker to filter inappropriate content. (bool)


## SeedanceV1ProImageToVideo

Generate videos from images using ByteDance's Seedance V1 Pro model via Kie.ai.
kie, seedance, bytedance, video generation, ai, image-to-video, pro, high-quality

Seedance V1 Pro offers high-quality image-to-video generation with improved
fidelity, transforming reference images into professional video sequences.

Use cases:
- Generate high-quality videos from images
- Create professional image animations
- Cinematic image-to-video conversion
- High-fidelity video production from images

**Tags:** 

**Fields:**
- **image**: The reference image to animate into a video. (ImageRef)
- **prompt**: Optional text to guide the video generation. (str)
- **resolution**: Video resolution. (Resolution)
- **duration**: Video duration in seconds. (int)
- **camera_fixed**: Whether to keep the camera fixed or allow camera movement. (bool)
- **seed**: Random seed for reproducible results. Use -1 for random seed. (int)
- **enable_safety_checker**: Enable safety checker to filter inappropriate content. (bool)


## SeedanceV1ProTextToVideo

Generate videos from text using ByteDance's Seedance V1 Pro model via Kie.ai.
kie, seedance, bytedance, video generation, ai, text-to-video, pro, high-quality

Seedance V1 Pro offers high-quality text-to-video generation with improved
fidelity, supporting multi-shot videos and cinematic scene transitions.

Use cases:
- Generate high-quality videos from text
- Create professional multi-shot videos
- Cinematic content creation
- High-fidelity video production

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the video to generate. Supports shot descriptions with [Cut to] notation. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **resolution**: Video resolution. (Resolution)
- **duration**: Video duration in seconds. (int)
- **camera_fixed**: Whether to keep the camera fixed or allow camera movement. (bool)
- **seed**: Random seed for reproducible results. Use -1 for random seed. (int)
- **enable_safety_checker**: Enable safety checker to filter inappropriate content. (bool)


## Sora2ProImageToVideo

Generate videos from images using OpenAI's Sora 2 Pro model via Kie.ai.
kie, sora, openai, video generation, ai, image-to-video, pro, 1080p

Sora 2 Pro Image-to-Video transforms reference images into professional-grade
HD videos with improved realism in motion and physics. Supports higher resolution
(HD) and longer clips (15s).

Use cases:
- Generate professional-grade HD videos from images
- Create longer 15-second video clips
- High-fidelity motion and physics from images
- Premium image-to-video content creation

**Tags:** 

**Fields:**
- **image**: The reference image to animate into a video. (ImageRef)
- **prompt**: Optional text to guide the video generation. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **n_frames**: Number of frames for the video. (int)
- **remove_watermark**: Whether to remove the watermark from the generated video. (bool)


## Sora2ProStoryboard

Generate storyboard videos using OpenAI's Sora 2 Pro model via Kie.ai.
kie, sora, openai, video generation, ai, storyboard, pro, 1080p

Sora 2 Pro Storyboard creates professional-grade storyboard videos with
scene transitions, cinematic framing, and professional editing techniques.

Use cases:
- Generate cinematic storyboards from text
- Create professional video previews
- Plan video shoots with visual storyboards
- Develop video concepts with scene sequences

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the storyboard sequence. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **n_frames**: Number of frames for the video. (int)
- **remove_watermark**: Whether to remove the watermark from the generated video. (bool)


## Sora2ProTextToVideo

Generate videos from text using OpenAI's Sora 2 Pro model via Kie.ai.
kie, sora, openai, video generation, ai, text-to-video, pro, 1080p

Sora 2 Pro Text-to-Video generates professional-grade HD videos from text prompts
with improved realism in motion and physics. Supports higher resolution (HD) and
longer clips (15s).

Use cases:
- Generate professional-grade HD videos from text
- Create longer 15-second clips
- High-fidelity motion and physics
- Premium text-to-video content creation

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the video to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **n_frames**: Number of frames for the video. (int)
- **remove_watermark**: Whether to remove the watermark from the generated video. (bool)


## Sora2TextToVideo

Generate videos from text using OpenAI's Sora 2 model via Kie.ai.
kie, sora, openai, video generation, ai, text-to-video, realistic

Sora 2 Text-to-Video generates short videos (up to 10 seconds) from text prompts,
emphasizing realistic motion, physics consistency, and native audio. Supports
standard and pro modes for different quality/speed tradeoffs.

Use cases:
- Generate realistic videos from text
- Create videos with native audio (dialogue/ambient sound)
- Quick text-to-video prototyping
- Cinematic content creation

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the video to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated video. (AspectRatio)
- **n_frames**: Number of frames for the video. (int)
- **remove_watermark**: Whether to remove the watermark from the generated video. (bool)
- **mode**: Generation mode: 'standard' or 'pro' for higher quality. (Mode)


## TopazVideoUpscale

Upscale and enhance videos using Topaz Labs AI via Kie.ai.
kie, topaz, upscale, enhance, video, ai, 4k

Uses Topaz Labs' video enhancement AI to upscale videos to 1080p or 4K,
sharpen frames, reduce noise, and interpolate frames for smoother motion.

Use cases:
- Upscale old or low-quality footage to HD/4K
- Enhance YouTube content or game recordings
- Restore and improve home videos
- Prepare videos for high-resolution displays

**Tags:** 

**Fields:**
- **video**: The video to upscale. (VideoRef)
- **resolution**: Target resolution for upscaling. (Resolution)
- **denoise**: Apply denoising to reduce artifacts. (bool)


## Flux2FlexImageToImage

Generate images using Black Forest Labs' Flux 2 Flex Image-to-Image model via Kie.ai.
kie, flux, flux-2, flux-flex, black-forest-labs, image generation, ai, image-to-image

Use cases:
- Transform existing images with text prompts
- Apply artistic styles to photos
- Create variations of existing images
- Enhance and modify images

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing how to transform the image. (str)
- **image**: The source image to transform. (ImageRef)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)
- **resolution**: Output image resolution. (Resolution)
- **steps**: Number of inference steps. Higher values may produce better quality but take longer. (int)
- **guidance_scale**: Guidance scale for the generation. Higher values adhere more closely to the prompt. (float)


## Flux2FlexTextToImage

Generate images using Black Forest Labs' Flux 2 Flex Text-to-Image model via Kie.ai.
kie, flux, flux-2, flux-flex, black-forest-labs, image generation, ai, text-to-image

Use cases:
- Generate high-quality images from text with flexible parameters
- Create professional visual content
- Generate images with fine detail and artistic style

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)
- **resolution**: Output image resolution. (Resolution)
- **steps**: Number of inference steps. Higher values may produce better quality but take longer. (int)
- **guidance_scale**: Guidance scale for the generation. Higher values adhere more closely to the prompt. (float)


## Flux2ProImageToImage

Generate images using Black Forest Labs' Flux 2 Pro Image-to-Image model via Kie.ai.
kie, flux, flux-2, flux-pro, black-forest-labs, image generation, ai, image-to-image

Use cases:
- Transform existing images with text prompts
- Apply artistic styles to photos
- Create variations of existing images
- Enhance and modify images

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing how to transform the image. (str)
- **image**: The source image to transform. (ImageRef)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)
- **resolution**: Output image resolution. (Resolution)
- **steps**: Number of inference steps. Higher values may produce better quality but take longer. (int)
- **guidance_scale**: Guidance scale for the generation. Higher values adhere more closely to the prompt. (float)


## Flux2ProTextToImage

Generate images using Black Forest Labs' Flux 2 Pro Text-to-Image model via Kie.ai.
kie, flux, flux-2, flux-pro, black-forest-labs, image generation, ai, text-to-image

Use cases:
- Generate high-quality artistic images from text
- Create professional visual content
- Generate images with fine detail and artistic style

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)
- **resolution**: Output image resolution. (Resolution)
- **steps**: Number of inference steps. Higher values may produce better quality but take longer. (int)
- **guidance_scale**: Guidance scale for the generation. Higher values adhere more closely to the prompt. (float)


## FluxKontext

Generate images using Black Forest Labs' Flux Kontext model via Kie.ai.
kie, flux, flux-kontext, black-forest-labs, image generation, ai, text-to-image, editing

Flux Kontext supports Pro (speed-optimized) and Max (quality-focused) variants
with features like multiple aspect ratios, safety controls, and async processing.

Use cases:
- Generate high-quality artistic images
- Advanced image editing and generation
- Create professional visual content
- Generate images with fine detail and artistic style

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)
- **mode**: Generation mode: 'pro' for speed, 'max' for quality. (Mode)


## GrokImagineTextToImage

Generate images using xAI's Grok Imagine Text-to-Image model via Kie.ai.
kie, grok, xai, image generation, ai, text-to-image, multimodal

Grok Imagine is a multimodal generative model that can generate images
from text prompts.

Use cases:
- Generate images from text descriptions
- Create visual content with AI

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)


## GrokImagineUpscale

Upscale images using xAI's Grok Imagine Upscale model via Kie.ai.
kie, grok, xai, upscale, enhance, image, ai, super-resolution

Grok Imagine Upscale enhances and upscales images to higher resolutions
while maintaining quality and detail.

Use cases:
- Upscale low-resolution images
- Enhance image quality and detail
- Improve resolution for printing or display

**Tags:** 

**Fields:**
- **image**: The image to upscale. (ImageRef)
- **scale_factor**: The upscaling factor (2x, 4x, or 8x). (ScaleFactor)


## NanoBanana

Generate images using Google's Nano Banana model (Gemini 2.5) via Kie.ai.
kie, nano-banana, google, gemini, image generation, ai, text-to-image, fast

Nano Banana is powered by Gemini 2.5 Flash Image and offers fast
language-driven image generation/editing, focusing on speed and iteration efficiency.

Use cases:
- Generate images with efficient processing
- Create visual content quickly
- Generate images for rapid prototyping
- Fast iteration on image concepts

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)


## NanoBananaPro

Generate images using Google's Nano Banana Pro model (Gemini 3.0) via Kie.ai.
kie, nano-banana-pro, google, gemini, image generation, ai, text-to-image, 4k, high-fidelity

Nano Banana Pro is based on Gemini 3.0 Pro Image and provides higher fidelity
with sharper structure, 4K output, and better text rendering for photorealistic results.

Use cases:
- Generate high-fidelity photorealistic images
- Create 4K resolution content
- Generate images with sharp text rendering
- Professional-grade visual content

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)


## QwenImageToImage

Transform images using Qwen's Image-to-Image model via Kie.ai.
kie, qwen, alibaba, image transformation, ai, image-to-image

Qwen's image-to-image model transforms existing images based on text prompts.

Use cases:
- Transform and edit existing images
- Apply styles and effects to photos
- Create variations of images
- Enhance or modify image content

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing how to transform the image. (str)
- **image**: The source image to transform. (ImageRef)
- **aspect_ratio**: The aspect ratio of the output image. (AspectRatio)


## QwenTextToImage

Generate images using Qwen's Text-to-Image model via Kie.ai.
kie, qwen, alibaba, image generation, ai, text-to-image

Qwen's text-to-image model generates high-quality images from text descriptions.

Use cases:
- Generate images from text descriptions
- Create artistic and realistic images
- Generate illustrations and artwork

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)


## Seedream45Edit

Edit images using ByteDance's Seedream 4.5 Edit model via Kie.ai.
kie, seedream, bytedance, image editing, ai, image-to-image, 4k

Seedream 4.5 Edit allows you to modify existing images while maintaining
high quality and detail fidelity up to 4K resolution.

Use cases:
- Edit and enhance existing images
- Apply style changes to photos
- Modify specific regions of images
- Improve image quality and resolution

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing how to edit the image. (str)
- **image**: The source image to edit. (ImageRef)
- **aspect_ratio**: The aspect ratio of the output image. (AspectRatio)


## Seedream45TextToImage

Generate images using ByteDance's Seedream 4.5 Text-to-Image model via Kie.ai.
kie, seedream, bytedance, image generation, ai, text-to-image, 4k

Seedream 4.5 generates high-quality visuals up to 4K resolution with
improved detail fidelity, multi-image blending, and sharp text/face rendering.

Use cases:
- Generate creative and artistic images from text
- Create diverse visual content up to 4K
- Generate illustrations with unique styles

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)


## TopazImageUpscale

Upscale and enhance images using Topaz Labs AI via Kie.ai.
kie, topaz, upscale, enhance, image, ai, super-resolution

Leverages Topaz Labs' image super-resolution models to upscale and enhance images.
Can enlarge images by 2x, 4x, etc., while unblurring and sharpening details.

Use cases:
- Upscale low-resolution images to high resolution
- Enhance old or degraded photos
- Improve AI-generated art quality
- Prepare images for large format printing

**Tags:** 

**Fields:**
- **image**: The image to upscale. (ImageRef)
- **scale_factor**: The upscaling factor (2x or 4x). (ScaleFactor)


## ZImage

Generate images using Alibaba's Z-Image Turbo model via Kie.ai.
kie, z-image, zimage, alibaba, image generation, ai, text-to-image, photorealistic

Z-Image Turbo produces realistic, detail-rich images with very low latency.
It supports bilingual text (English/Chinese) in images with sharp text rendering.

Use cases:
- Generate high-quality photorealistic images quickly
- Create images with embedded text (English/Chinese)
- Generate detailed illustrations with low latency
- Product visualizations

**Tags:** 

**Fields:**
- **prompt**: The text prompt describing the image to generate. (str)
- **aspect_ratio**: The aspect ratio of the generated image. (AspectRatio)


## ExtractImages

Extract images from a PDF file.

Use cases:
- Extract embedded images from PDF documents
- Save PDF images as separate files
- Process PDF images for analysis

**Tags:** pdf, image, extract

**Fields:**
- **pdf**: The PDF file to extract images from (DocumentRef)
- **start_page**: The start page to extract (int)
- **end_page**: The end page to extract (int)


## ExtractPageMetadata

Extract metadata from PDF pages like dimensions, rotation, etc.

Use cases:
- Analyze page layouts
- Get page dimensions
- Check page orientations

**Tags:** pdf, metadata, pages

**Fields:**
- **pdf**: The PDF file to analyze (DocumentRef)
- **start_page**: The start page to extract. 0-based indexing (int)
- **end_page**: The end page to extract. -1 for all pages (int)


## ExtractTables

Extract tables from a PDF file into dataframes.

Use cases:
- Extract tabular data from PDF documents
- Convert PDF tables to structured data formats
- Process PDF tables for analysis
- Import PDF reports into data analysis pipelines

**Tags:** pdf, tables, dataframe, extract

**Fields:**
- **pdf**: The PDF document to extract tables from (DocumentRef)
- **start_page**: First page to extract tables from (0-based, None for first page) (int)
- **end_page**: Last page to extract tables from (0-based, None for last page) (int)
- **table_settings**: Settings for table extraction algorithm (dict)


## ExtractText

Extract text content from a PDF file.

Use cases:
- Convert PDF documents to plain text
- Extract content for analysis
- Enable text search in PDF documents

**Tags:** pdf, text, extract

**Fields:**
- **pdf**: The PDF file to extract text from (DocumentRef)
- **start_page**: The start page to extract. 0-based indexing (int)
- **end_page**: The end page to extract. -1 for all pages (int)


## GetPageCount

Get the total number of pages in a PDF file.

Use cases:
- Check document length
- Plan batch processing

**Tags:** pdf, pages, count

**Fields:**
- **pdf**: The PDF file to analyze (DocumentRef)


## ConvertToMarkdown

Converts various document formats to markdown using MarkItDown.

Use cases:
- Convert Word documents to markdown
- Convert Excel files to markdown tables
- Convert PowerPoint to markdown content

**Tags:** markdown, convert, document

**Fields:**
- **document**: The document to convert to markdown (DocumentRef)


## Delete

Delete records from a Supabase table.

**Tags:** supabase, database, delete, remove

**Fields:**
- **table_name**: Table to delete from (str)
- **filters**: Filters to select rows to delete (required for safety) (list[tuple[str, nodetool.nodes.lib.supabase.FilterOp, typing.Any]])


## Insert

Insert record(s) into a Supabase table.

**Tags:** supabase, database, insert, add, record

**Fields:**
- **table_name**: Table to insert into (str)
- **records**: One or multiple rows to insert (list[dict[str, typing.Any]] | dict[str, typing.Any])
- **return_rows**: Return inserted rows (uses select('*')) (bool)


## RPC

Call a PostgreSQL function via Supabase RPC.

**Tags:** supabase, database, rpc, function

**Fields:**
- **function**: RPC function name (str)
- **params**: Function params (dict[str, typing.Any])
- **to_dataframe**: Return DataframeRef if result is a list of records (bool)


## Select

Query records from a Supabase table.

**Tags:** supabase, database, query, select

**Fields:**
- **table_name**: Table to query (str)
- **columns**: Columns to select (RecordType)
- **filters**: List of typed filters to apply (list[tuple[str, nodetool.nodes.lib.supabase.FilterOp, typing.Any]])
- **order_by**: Column to order by (str)
- **descending**: Order direction (bool)
- **limit**: Max rows to return (0 = no limit) (int)
- **to_dataframe**: Return a DataframeRef instead of list of dicts (bool)


## Update

Update records in a Supabase table.

**Tags:** supabase, database, update, modify, change

**Fields:**
- **table_name**: Table to update (str)
- **values**: New values (dict[str, typing.Any])
- **filters**: Filters to select rows to update (list[tuple[str, nodetool.nodes.lib.supabase.FilterOp, typing.Any]])
- **return_rows**: Return updated rows (uses select('*')) (bool)


## Upsert

Insert or update (upsert) records in a Supabase table.

**Tags:** supabase, database, upsert, merge

**Fields:**
- **table_name**: Table to upsert into (str)
- **records**: One or multiple rows to upsert (list[dict[str, typing.Any]] | dict[str, typing.Any])
- **return_rows**: Return upserted rows (uses select('*')) (bool)


## AddLabel

Adds a label to a Gmail message.

**Tags:** email, gmail, label

**Fields:**
- **message_id**: Message ID to label (str)
- **label**: Label to add to the message (str)


## EmailFields

Decomposes an email into its individual components.

Takes an Email object and returns its individual fields:
- id: Message ID
- subject: Email subject
- sender: Sender address
- date: Datetime of email
- body: Email body content

**Tags:** email, decompose, extract

**Fields:**
- **email**: Email object to decompose (Email)


## GmailSearch

Searches Gmail using Gmail-specific search operators and yields matching emails.

Use cases:
- Search for emails based on specific criteria
- Retrieve emails from a specific sender
- Filter emails by subject, sender, or date

**Tags:** email, gmail, search

**Fields:**
- **from_address**: Sender's email address to search for (str)
- **to_address**: Recipient's email address to search for (str)
- **subject**: Text to search for in email subject (str)
- **body**: Text to search for in email body (str)
- **date_filter**: Date filter to search for (DateFilter)
- **keywords**: Custom keywords or labels to search for (str)
- **folder**: Email folder to search in (GmailFolder)
- **text**: General text to search for anywhere in the email (str)
- **max_results**: Maximum number of emails to return (int)
- **retry_attempts**: Maximum retry attempts for Gmail operations (int)
- **retry_base_delay**: Base delay (seconds) for exponential backoff (float)
- **retry_max_delay**: Maximum delay (seconds) for exponential backoff (float)
- **retry_factor**: Exponential growth factor for backoff (float)
- **retry_jitter**: Random jitter (seconds) added to each backoff (float)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.mail.GmailSearch.OutputType, NoneType]


## MoveToArchive

Moves specified emails to Gmail archive.

**Tags:** email, gmail, archive

**Fields:**
- **message_id**: Message ID to archive (str)


## SendEmail

Send a plain text email via SMTP.

Use cases:
- Send simple notification messages
- Automate email reports

**Tags:** email, smtp, send

**Fields:**
- **smtp_server**: SMTP server hostname (str)
- **smtp_port**: SMTP server port (int)
- **username**: SMTP username (str)
- **password**: SMTP password (str)
- **from_address**: Sender email address (str)
- **to_address**: Recipient email address (str)
- **subject**: Email subject (str)
- **body**: Email body (str)
- **retry_attempts**: Maximum retry attempts for SMTP send (int)
- **retry_base_delay**: Base delay (seconds) for exponential backoff (float)
- **retry_max_delay**: Maximum delay (seconds) for exponential backoff (float)
- **retry_factor**: Exponential growth factor for backoff (float)
- **retry_jitter**: Random jitter (seconds) added to each backoff (float)


## ExtractMarkdown

Convert PDF to Markdown format using pymupdf4llm.

Use cases:
- Convert PDF documents to markdown format
- Preserve document structure in markdown
- Create editable markdown from PDFs

**Tags:** pdf, markdown, convert

**Fields:**
- **pdf**: The PDF document to convert to markdown (DocumentRef)
- **start_page**: First page to extract (0-based index) (int)
- **end_page**: Last page to extract (-1 for last page) (int)


## ExtractTables

Extract tables from a PDF document using PyMuPDF.

Use cases:
- Extract tabular data from PDFs
- Convert PDF tables to structured formats
- Analyze table layouts and content

**Tags:** pdf, tables, extract, structured

**Fields:**
- **pdf**: The PDF document to extract tables from (DocumentRef)
- **start_page**: First page to extract (0-based index) (int)
- **end_page**: Last page to extract (-1 for last page) (int)


## ExtractText

Extract plain text from a PDF document using PyMuPDF.

Use cases:
- Extract raw text content from PDFs
- Convert PDF documents to plain text
- Prepare text for further processing

**Tags:** pdf, text, extract

**Fields:**
- **pdf**: The PDF document to extract text from (DocumentRef)
- **start_page**: First page to extract (0-based index) (int)
- **end_page**: Last page to extract (-1 for last page) (int)


## ExtractTextBlocks

Extract text blocks with their bounding boxes from a PDF.

Use cases:
- Analyze text layout and structure
- Extract text while preserving block-level formatting
- Get text position information

**Tags:** pdf, text, blocks, layout

**Fields:**
- **pdf**: The PDF document to extract text blocks from (DocumentRef)
- **start_page**: First page to extract (0-based index) (int)
- **end_page**: Last page to extract (-1 for last page) (int)


## ExtractTextWithStyle

Extract text with style information (font, size, color) from a PDF.

Use cases:
- Preserve text formatting during extraction
- Analyze document styling
- Extract text with font information

**Tags:** pdf, text, style, formatting

**Fields:**
- **pdf**: The PDF document to extract styled text from (DocumentRef)
- **start_page**: First page to extract (0-based index) (int)
- **end_page**: Last page to extract (-1 for last page) (int)


## CircleNode

Generate SVG circle element with customizable position, radius, and styling.

Use cases:
- Create circular shapes and icons
- Design buttons, badges, and indicators
- Build data visualizations like pie charts
- Create decorative elements and patterns

**Tags:** svg, shape, vector, circle

**Fields:**
- **cx**: Center X coordinate (int)
- **cy**: Center Y coordinate (int)
- **radius**: Radius (int)
- **fill**: Fill color (ColorRef)
- **stroke**: Stroke color (ColorRef)
- **stroke_width**: Stroke width (int)


## ClipPath

Create clipping paths for SVG elements.

Use cases:
- Mask parts of elements
- Create complex shapes through clipping
- Apply visual effects using masks

**Tags:** svg, clip, mask

**Fields:**
- **clip_content**: SVG element to use as clip path (SVGElement)
- **content**: SVG element to clip (SVGElement)


## Document

Combine SVG elements into a complete SVG document.

Use cases:
- Combine multiple SVG elements into a single document
- Set document-level properties like viewBox and dimensions
- Export complete SVG documents

**Tags:** svg, document, combine

**Fields:**
- **content**: SVG content (str | nodetool.metadata.types.SVGElement | list[nodetool.metadata.types.SVGElement])
- **width**: Document width (int)
- **height**: Document height (int)
- **viewBox**: SVG viewBox attribute (str)


## DropShadow

Apply drop shadow filter effect to SVG elements for depth.

Use cases:
- Add depth and elevation to elements
- Create realistic shadow effects
- Enhance visual hierarchy
- Improve element separation and readability

**Tags:** svg, filter, shadow, effects

**Fields:**
- **std_deviation**: Standard deviation for blur (float)
- **dx**: X offset for shadow (int)
- **dy**: Y offset for shadow (int)
- **color**: Color for shadow (ColorRef)


## EllipseNode

Generate SVG ellipse element with customizable position, radii, and styling.

Use cases:
- Create oval shapes and organic forms
- Design speech bubbles and callouts
- Build data visualization elements
- Create decorative patterns and borders

**Tags:** svg, shape, vector, ellipse

**Fields:**
- **cx**: Center X coordinate (int)
- **cy**: Center Y coordinate (int)
- **rx**: X radius (int)
- **ry**: Y radius (int)
- **fill**: Fill color (ColorRef)
- **stroke**: Stroke color (ColorRef)
- **stroke_width**: Stroke width (int)


## GaussianBlur

Apply Gaussian blur filter effect to SVG elements.

Use cases:
- Create soft focus and depth effects
- Add subtle shadows and glows
- Simulate motion blur
- Soften edges in graphics

**Tags:** svg, filter, blur, effects

**Fields:**
- **std_deviation**: Standard deviation for blur (float)


## Gradient

Create linear or radial gradients for SVG elements.

Use cases:
- Add smooth color transitions
- Create complex color effects
- Define reusable gradient definitions

**Tags:** svg, gradient, color

**Fields:**
- **gradient_type**: Type of gradient (GradientType)
- **x1**: Start X position (linear) or center X (radial) (float)
- **y1**: Start Y position (linear) or center Y (radial) (float)
- **x2**: End X position (linear) or radius X (radial) (float)
- **y2**: End Y position (linear) or radius Y (radial) (float)
- **color1**: Start color of gradient (ColorRef)
- **color2**: End color of gradient (ColorRef)


## LineNode

Generate SVG line element with customizable endpoints and styling.

Use cases:
- Draw straight lines and connectors
- Create dividers and separators
- Build diagrams and flowcharts
- Design grid patterns and borders

**Tags:** svg, shape, vector, line

**Fields:**
- **x1**: Start X coordinate (int)
- **y1**: Start Y coordinate (int)
- **x2**: End X coordinate (int)
- **y2**: End Y coordinate (int)
- **stroke**: Stroke color (ColorRef)
- **stroke_width**: Stroke width (int)


## PathNode

Generate SVG path element using path data commands.

Use cases:
- Create complex curved and custom shapes
- Build logos and custom icons
- Design intricate patterns and illustrations
- Import path data from design tools

**Tags:** svg, shape, vector, path

**Fields:**
- **path_data**: SVG path data (d attribute) (str)
- **fill**: Fill color (ColorRef)
- **stroke**: Stroke color (ColorRef)
- **stroke_width**: Stroke width (int)


## PolygonNode

Generate SVG polygon element with multiple vertices.

Use cases:
- Create multi-sided shapes like triangles, pentagons, stars
- Build custom icons and symbols
- Design complex geometric patterns
- Create irregular shapes and forms

**Tags:** svg, shape, vector, polygon

**Fields:**
- **points**: Points in format 'x1,y1 x2,y2 x3,y3...' (str)
- **fill**: Fill color (ColorRef)
- **stroke**: Stroke color (ColorRef)
- **stroke_width**: Stroke width (int)


## RectNode

Generate SVG rectangle element with customizable position, size, and styling.

Use cases:
- Create rectangular shapes in SVG documents
- Design borders, frames, and backgrounds
- Build user interface components
- Create geometric patterns and layouts

**Tags:** svg, shape, vector, rectangle

**Fields:**
- **x**: X coordinate (int)
- **y**: Y coordinate (int)
- **width**: Width (int)
- **height**: Height (int)
- **fill**: Fill color (ColorRef)
- **stroke**: Stroke color (ColorRef)
- **stroke_width**: Stroke width (int)


## SVGToImage

Create an SVG document and convert it to a raster image in one step.

Use cases:
- Create and rasterize SVG documents in a single operation
- Generate image files from SVG elements
- Convert vector graphics to bitmap format with custom dimensions

**Tags:** svg, document, raster, convert

**Fields:**
- **content**: SVG content (str | nodetool.metadata.types.SVGElement | list[nodetool.metadata.types.SVGElement])
- **width**: Document width (int)
- **height**: Document height (int)
- **viewBox**: SVG viewBox attribute (str)
- **scale**: Scale factor for rasterization (int)


## Text

Add text elements to SVG.

Use cases:
- Add labels to vector graphics
- Create text-based logos
- Generate dynamic text content in SVGs

**Tags:** svg, text, typography

**Fields:**
- **text**: Text content (str)
- **x**: X coordinate (int)
- **y**: Y coordinate (int)
- **font_family**: Font family (str)
- **font_size**: Font size (int)
- **fill**: Text color (ColorRef)
- **text_anchor**: Text anchor position (SVGTextAnchor)


## Transform

Apply transformations to SVG elements.

Use cases:
- Rotate, scale, or translate elements
- Create complex transformations
- Prepare elements for animation

**Tags:** svg, transform, animation

**Fields:**
- **content**: SVG element to transform (SVGElement)
- **translate_x**: X translation (float)
- **translate_y**: Y translation (float)
- **rotate**: Rotation angle in degrees (float)
- **scale_x**: X scale factor (float)
- **scale_y**: Y scale factor (float)


## ConvertFile

Converts between different document formats using pandoc.

Use cases:
- Convert between various document formats (Markdown, HTML, LaTeX, etc.)
- Generate documentation in different formats
- Create publication-ready documents

**Tags:** convert, document, format, pandoc

**Fields:**
- **input_path**: Path to the input file (FilePath)
- **input_format**: Input format (InputFormat)
- **output_format**: Output format (OutputFormat)
- **extra_args**: Additional pandoc arguments (list[str])


## ConvertText

Converts text content between different document formats using pandoc.

Use cases:
- Convert text content between various formats (Markdown, HTML, LaTeX, etc.)
- Transform content without saving to disk
- Process text snippets in different formats

**Tags:** convert, text, format, pandoc

**Fields:**
- **content**: Text content to convert (str)
- **input_format**: Input format (InputFormat)
- **output_format**: Output format (OutputFormat)
- **extra_args**: Additional pandoc arguments (list[str])


## ExtractFeedMetadata

Extracts metadata from an RSS feed.

Use cases:
- Get feed information
- Validate feed details
- Extract feed metadata

**Tags:** rss, metadata, feed

**Fields:**
- **url**: URL of the RSS feed (str)


## FetchRSSFeed

Fetches and parses an RSS feed from a URL.

Use cases:
- Monitor news feeds
- Aggregate content from multiple sources
- Process blog updates

**Tags:** rss, feed, network

**Fields:**
- **url**: URL of the RSS feed to fetch (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.rss.FetchRSSFeed.OutputType, NoneType]


## AbsolutePath

Return the absolute path of a file or directory.

Use cases:
- Convert relative paths to absolute
- Get full system path
- Resolve path references

**Tags:** files, path, absolute

**Fields:**
- **path**: Path to convert to absolute (str)


## AccessedTime

Get file last accessed timestamp.

**Tags:** files, metadata, accessed, time

**Fields:**
- **path**: Path to file (str)


## Basename

Get the base name component of a file path.

Use cases:
- Extract filename from full path
- Get file name without directory
- Process file names independently

**Tags:** files, path, basename

**Fields:**
- **path**: File path to get basename from (str)
- **remove_extension**: Remove file extension from basename (bool)


## CopyFile

Copy a file from source to destination path.

Use cases:
- Create file backups
- Duplicate files for processing
- Copy files to new locations

**Tags:** files, copy, manage

**Fields:**
- **source_path**: Source file path (str)
- **destination_path**: Destination file path (str)


## CreateDirectory

Create a new directory at specified path.

Use cases:
- Set up directory structure for file organization
- Create output directories for processed files

**Tags:** files, directory, create

**Fields:**
- **path**: Directory path to create (str)
- **exist_ok**: Don't error if directory already exists (bool)


## CreatedTime

Get file creation timestamp.

**Tags:** files, metadata, created, time

**Fields:**
- **path**: Path to file (str)


## Dirname

Get the directory name component of a file path.

Use cases:
- Extract directory path from full path
- Get parent directory
- Process directory paths

**Tags:** files, path, dirname

**Fields:**
- **path**: File path to get dirname from (str)


## FileExists

Check if a file or directory exists at the specified path.

Use cases:
- Validate file presence before processing
- Implement conditional logic based on file existence

**Tags:** files, check, exists

**Fields:**
- **path**: Path to check for existence (str)


## FileExtension

Get file extension.

**Tags:** files, metadata, extension

**Fields:**
- **path**: Path to file (str)


## FileName

Get file name without path.

**Tags:** files, metadata, name

**Fields:**
- **path**: Path to file (str)


## FileNameMatch

Match a filename against a pattern using Unix shell-style wildcards.

Use cases:
- Filter files by name pattern
- Validate file naming conventions
- Match file extensions

**Tags:** files, pattern, match, filter

**Fields:**
- **filename**: Filename to check (str)
- **pattern**: Pattern to match against (e.g. *.txt, data_*.csv) (str)
- **case_sensitive**: Whether the pattern matching should be case-sensitive (bool)


## FilterFileNames

Filter a list of filenames using Unix shell-style wildcards.

Use cases:
- Filter multiple files by pattern
- Batch process files matching criteria
- Select files by extension

**Tags:** files, pattern, filter, list

**Fields:**
- **filenames**: list of filenames to filter (list[str])
- **pattern**: Pattern to filter by (e.g. *.txt, data_*.csv) (str)
- **case_sensitive**: Whether the pattern matching should be case-sensitive (bool)


## GetDirectory

Get directory containing the file.

**Tags:** files, metadata, directory

**Fields:**
- **path**: Path to file (str)


## GetFileSize

Get file size in bytes.

**Tags:** files, metadata, size

**Fields:**
- **path**: Path to file (str)


## GetPathInfo

Gets information about a path.

Use cases:
- Extract path components
- Parse file paths

**Tags:** path, info, metadata

**Fields:**
- **path**: Path to analyze (str)


## IsDirectory

Check if path is a directory.

**Tags:** files, metadata, type

**Fields:**
- **path**: Path to check (str)


## IsFile

Check if path is a file.

**Tags:** files, metadata, type

**Fields:**
- **path**: Path to check (str)


## JoinPaths

Joins path components.

Use cases:
- Build file paths
- Create cross-platform paths

**Tags:** path, join, combine

**Fields:**
- **paths**: Path components to join (list[str])


## ListFiles

list files in a directory matching a pattern.

Use cases:
- Get files for batch processing
- Filter files by extension or pattern

**Tags:** files, list, directory

**Fields:**
- **folder**: Directory to scan (str)
- **pattern**: File pattern to match (e.g. *.txt) (str)
- **include_subdirectories**: Search subdirectories (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.os.ListFiles.OutputType, NoneType]


## ModifiedTime

Get file last modified timestamp.

**Tags:** files, metadata, modified, time

**Fields:**
- **path**: Path to file (str)


## MoveFile

Move a file from source to destination path.

Use cases:
- Organize files into directories
- Process and archive files
- Relocate completed files

**Tags:** files, move, manage

**Fields:**
- **source_path**: Source file path (str)
- **destination_path**: Destination file path (str)


## NormalizePath

Normalizes a path.

Use cases:
- Standardize paths
- Remove redundant separators

**Tags:** path, normalize, clean

**Fields:**
- **path**: Path to normalize (str)


## OpenWorkspaceDirectory

Open the workspace directory.

**Tags:** files, workspace, directory

**Fields:**


## PathToString

Convert a FilePath object to a string.

Use cases:
- Get raw string path from FilePath object
- Convert FilePath for string operations
- Extract path string for external use

**Tags:** files, path, string, convert

**Fields:**
- **file_path**: File path to convert to string (str)


## RelativePath

Return a relative path to a target from a start directory.

Use cases:
- Create relative path references
- Generate portable paths
- Compare file locations

**Tags:** files, path, relative

**Fields:**
- **target_path**: Target path to convert to relative (str)
- **start_path**: Start path for relative conversion (str)


## ShowNotification

Shows a system notification.

Use cases:
- Alert user of completed tasks
- Show process status
- Display important messages

**Tags:** notification, system, alert

**Fields:**
- **title**: Title of the notification (str)
- **message**: Content of the notification (str)
- **timeout**: How long the notification should stay visible (in seconds) (int)


## SplitExtension

Split a path into root and extension components.

Use cases:
- Extract file extension
- Process filename without extension
- Handle file types

**Tags:** files, path, extension, split

**Fields:**
- **path**: Path to split (str)


## SplitPath

Split a path into directory and file components.

Use cases:
- Separate directory from filename
- Process path components separately
- Extract path parts

**Tags:** files, path, split

**Fields:**
- **path**: Path to split (str)


## WorkspaceDirectory

Get the workspace directory.

**Tags:** files, workspace, directory

**Fields:**


## CreateTable

Create a new SQLite table with specified columns.

Use cases:
- Initialize database schema for flashcards
- Set up tables for persistent storage
- Create memory structures for agents

**Tags:** sqlite, database, table, create, schema

**Fields:**
- **database_name**: Name of the SQLite database file (str)
- **table_name**: Name of the table to create (str)
- **columns**: Column definitions (RecordType)
- **add_primary_key**: Automatically make first integer column PRIMARY KEY AUTOINCREMENT (bool)
- **if_not_exists**: Only create table if it doesn't exist (bool)


## Delete

Delete records from a SQLite table.

Use cases:
- Remove flashcards
- Delete agent memory
- Clean up old data

**Tags:** sqlite, database, delete, remove, drop

**Fields:**
- **database_name**: Name of the SQLite database file (str)
- **table_name**: Name of the table to delete from (str)
- **where**: WHERE clause (without 'WHERE' keyword), e.g., 'id = 1'. REQUIRED for safety. (str)


## ExecuteSQL

Execute arbitrary SQL statements for advanced operations.

Use cases:
- Complex queries with joins
- Aggregate functions (COUNT, SUM, AVG)
- Custom SQL operations

**Tags:** sqlite, database, sql, execute, custom

**Fields:**
- **database_name**: Name of the SQLite database file (str)
- **sql**: SQL statement to execute (str)
- **parameters**: Parameters for parameterized queries (use ? in SQL) (list[typing.Any])


## GetDatabasePath

Get the full path to a SQLite database file.

Use cases:
- Reference database location
- Verify database exists
- Pass path to external tools

**Tags:** sqlite, database, path, location

**Fields:**
- **database_name**: Name of the SQLite database file (str)


## Insert

Insert a record into a SQLite table.

Use cases:
- Add new flashcards to database
- Store agent observations
- Persist workflow results

**Tags:** sqlite, database, insert, add, record

**Fields:**
- **database_name**: Name of the SQLite database file (str)
- **table_name**: Name of the table to insert into (str)
- **data**: Data to insert as dict (column: value) (dict[str, typing.Any])


## Query

Query records from a SQLite table.

Use cases:
- Retrieve flashcards for review
- Search agent memory
- Fetch stored data

**Tags:** sqlite, database, query, select, search, retrieve

**Fields:**
- **database_name**: Name of the SQLite database file (str)
- **table_name**: Name of the table to query (str)
- **where**: WHERE clause (without 'WHERE' keyword), e.g., 'id = 1' (str)
- **columns**: Columns to select (RecordType)
- **order_by**: ORDER BY clause (without 'ORDER BY' keyword) (str)
- **limit**: Maximum number of rows to return (0 = no limit) (int)


## Update

Update records in a SQLite table.

Use cases:
- Update flashcard content
- Modify stored records
- Change agent memory

**Tags:** sqlite, database, update, modify, change

**Fields:**
- **database_name**: Name of the SQLite database file (str)
- **table_name**: Name of the table to update (str)
- **data**: Data to update as dict (column: new_value) (dict[str, typing.Any])
- **where**: WHERE clause (without 'WHERE' keyword), e.g., 'id = 1' (str)


## Add

Adds two numbers together.

Use cases:
- Perform basic arithmetic operations
- Calculate totals and sums
- Combine numerical values
- Increment counters and scores

**Tags:** math, add, plus, +, sum

**Fields:**
- **a** (int | float)
- **b** (int | float)


## Cosine

Computes cosine of the given angle in radians.

**Tags:** math, cosine, trig

**Fields:**
- **angle_rad** (int | float)


## Divide

Divides A by B to calculate the quotient.

Use cases:
- Calculate averages and ratios
- Distribute quantities evenly
- Determine rates and proportions
- Compute per-unit values

**Tags:** math, divide, division, quotient, /

**Fields:**
- **a** (int | float)
- **b** (int | float)


## MathFunction

Performs a selected unary math operation on an input.

**Tags:** math, negate, absolute, square, cube, square_root, cube_root, sine, cosine, tangent, arcsine, arccosine, arctangent, log,   -, abs, ^2, ^3, sqrt, cbrt, sin, cos, tan, asin, acos, atan, log

**Fields:**
- **input** (int | float)
- **operation**: Unary operation to perform (Operation)


## Modulus

Computes A modulo B to find the remainder after division.

Use cases:
- Determine if numbers are even or odd
- Implement cyclic patterns and rotations
- Calculate remainders in division
- Build repeating sequences

**Tags:** math, modulus, modulo, remainder, %

**Fields:**
- **a** (int | float)
- **b** (int | float)


## Multiply

Multiplies two numbers together.

Use cases:
- Calculate products and totals
- Scale values by factors
- Compute areas and volumes
- Apply multipliers and rates

**Tags:** math, multiply, product, *, times

**Fields:**
- **a** (int | float)
- **b** (int | float)


## Power

Raises base to the given exponent.

**Tags:** math, power, exponent, ^

**Fields:**
- **base** (int | float)
- **exponent** (int | float)


## Sine

Computes sine of the given angle in radians.

**Tags:** math, sine, trig

**Fields:**
- **angle_rad** (int | float)


## Sqrt

Computes square root of x.

**Tags:** math, sqrt, square_root

**Fields:**
- **x** (int | float)


## Subtract

Subtracts B from A.

Use cases:
- Calculate differences between values
- Determine remaining amounts
- Compute offsets and deltas
- Track decrements and reductions

**Tags:** math, subtract, minus, -, difference

**Fields:**
- **a** (int | float)
- **b** (int | float)


## GetSecret

Get a secret value from configuration.

**Tags:** secrets, credentials, configuration

**Fields:**
- **name**: Secret key name (str)
- **default**: Default value if not found (str)


## ExtractBulletLists

Extracts bulleted lists from markdown.

Use cases:
- Extract unordered list items
- Analyze bullet point structures
- Convert bullet lists to structured data

**Tags:** markdown, lists, bullets, extraction

**Fields:**
- **markdown**: The markdown text to analyze (str)


## ExtractCodeBlocks

Extracts code blocks and their languages from markdown.

Use cases:
- Extract code samples for analysis
- Collect programming examples
- Analyze code snippets in documentation

**Tags:** markdown, code, extraction

**Fields:**
- **markdown**: The markdown text to analyze (str)


## ExtractHeaders

Extracts headers and creates a document structure/outline.

Use cases:
- Generate table of contents
- Analyze document structure
- Extract main topics from documents

**Tags:** markdown, headers, structure

**Fields:**
- **markdown**: The markdown text to analyze (str)
- **max_level**: Maximum header level to extract (1-6) (int)


## ExtractLinks

Extracts all links from markdown text.

Use cases:
- Extract references and citations from academic documents
- Build link graphs from markdown documentation
- Analyze external resources referenced in markdown files

**Tags:** markdown, links, extraction

**Fields:**
- **markdown**: The markdown text to analyze (str)
- **include_titles**: Whether to include link titles in output (bool)


## ExtractNumberedLists

Extracts numbered lists from markdown.

Use cases:
- Extract ordered list items
- Analyze enumerated structures
- Convert numbered lists to structured data

**Tags:** markdown, lists, numbered, extraction

**Fields:**
- **markdown**: The markdown text to analyze (str)


## ExtractTables

Extracts tables from markdown and converts them to structured data.

Use cases:
- Extract tabular data from markdown
- Convert markdown tables to structured formats
- Analyze tabulated information

**Tags:** markdown, tables, data

**Fields:**
- **markdown**: The markdown text to analyze (str)


## AddTimeDelta

Add or subtract time from a datetime using specified intervals.

Use cases:
- Calculate future/past dates
- Generate date ranges
- Schedule events at specific intervals
- Calculate expiration dates and deadlines

**Tags:** datetime, add, subtract, delta, offset

**Fields:**
- **input_datetime**: Starting datetime (Datetime)
- **days**: Number of days to add (negative to subtract) (int)
- **hours**: Number of hours to add (negative to subtract) (int)
- **minutes**: Number of minutes to add (negative to subtract) (int)


## BoundaryTime

Get the start or end boundary of a time period (day, week, month, year).

Use cases:
- Get period boundaries for reporting and analytics
- Normalize dates to period starts/ends
- Calculate billing cycles
- Group data by time periods

**Tags:** datetime, start, end, boundary, day, week, month, year

**Fields:**
- **input_datetime**: Input datetime (Datetime)
- **period**: Time period type (PeriodType)
- **boundary**: Start or end of period (BoundaryType)
- **start_monday**: For week period: Consider Monday as start of week (False for Sunday) (bool)


## DateDifference

Calculate the time difference between two datetimes.

Use cases:
- Calculate time periods between events
- Measure durations and elapsed time
- Track age or time since events
- Compute service level agreement (SLA) metrics

**Tags:** datetime, difference, duration, elapsed

**Fields:**
- **start_date**: Start datetime (Datetime)
- **end_date**: End datetime (Datetime)


## DateRange

Generate a list of dates between start and end dates with custom intervals.

Use cases:
- Generate date sequences for reporting
- Create date-based iterations in workflows
- Build calendar views
- Schedule recurring events

**Tags:** datetime, range, list, sequence

**Fields:**
- **start_date**: Start date of the range (Datetime)
- **end_date**: End date of the range (Datetime)
- **step_days**: Number of days between each date (int)


## DateToDatetime

Convert a Date object to a Datetime object at midnight.

Use cases:
- Convert dates to datetime for time calculations
- Standardize date types in workflows
- Prepare dates for timestamp comparisons
- Convert legacy date formats

**Tags:** date, datetime, convert, transformation

**Fields:**
- **input_date**: Date to convert (Date)


## DatetimeToDate

Convert a Datetime object to a Date object, removing time component.

Use cases:
- Extract date portion from timestamps
- Remove time information for date-only comparisons
- Normalize datetime data to dates
- Simplify date-based grouping

**Tags:** date, datetime, convert, transformation

**Fields:**
- **input_datetime**: Datetime to convert (Datetime)


## FormatDateTime

Convert a datetime object to a custom formatted string.

Use cases:
- Standardize date formats across systems
- Prepare dates for different locales and regions
- Generate human-readable date strings
- Format dates for filenames and reports

**Tags:** datetime, format, convert, string

**Fields:**
- **input_datetime**: Datetime object to format (Datetime)
- **output_format**: Desired output format (DateFormat)


## GetQuarter

Get the quarter number and start/end dates for a given datetime.

Use cases:
- Financial reporting periods
- Quarterly analytics and metrics
- Business cycle calculations
- Group data by fiscal quarters

**Tags:** datetime, quarter, period, fiscal

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## GetWeekday

Get the weekday name or number from a datetime.

Use cases:
- Get day names for scheduling and calendar displays
- Filter events by weekday
- Build day-of-week based logic
- Generate weekly reports

**Tags:** datetime, weekday, name, day

**Fields:**
- **input_datetime**: Input datetime (Datetime)
- **as_name**: Return weekday name instead of number (0-6) (bool)


## IsDateInRange

Check if a date falls within a specified range with optional inclusivity.

Use cases:
- Validate date ranges in forms and inputs
- Filter date-based data
- Check if events fall within specific periods
- Implement date-based access control

**Tags:** datetime, range, check, validate

**Fields:**
- **check_date**: Date to check (Datetime)
- **start_date**: Start of date range (Datetime)
- **end_date**: End of date range (Datetime)
- **inclusive**: Include start and end dates in range (bool)


## Now

Get the current date and time in UTC timezone.

Use cases:
- Generate timestamps for events and logs
- Set default datetime values in workflows
- Calculate time-based conditions
- Track real-time operations

**Tags:** datetime, current, now, timestamp

**Fields:**


## ParseDate

Parse a date string into a structured Date object.

Use cases:
- Convert date strings from various sources into standard format
- Extract date components from text input
- Validate and normalize date formats
- Process dates from CSV, JSON, or API responses

**Tags:** date, parse, format, convert

**Fields:**
- **date_string**: The date string to parse (str)
- **input_format**: Format of the input date string (DateFormat)


## ParseDateTime

Parse a date/time string into a structured Datetime object.

Use cases:
- Extract datetime components from strings
- Convert between datetime formats
- Process timestamps from logs and databases
- Standardize datetime data from multiple sources

**Tags:** datetime, parse, format, convert

**Fields:**
- **datetime_string**: The datetime string to parse (str)
- **input_format**: Format of the input datetime string (DateFormat)


## RelativeTime

Get datetime relative to current time (past or future) with configurable units.

Use cases:
- Calculate past or future dates dynamically
- Generate relative timestamps for scheduling
- Set expiration times
- Create time-based filters

**Tags:** datetime, past, future, relative, hours, days, months

**Fields:**
- **amount**: Amount of time units (int)
- **unit**: Time unit type (TimeUnitType)
- **direction**: Past or future (TimeDirection)


## Today

Get the current date in Date format.

Use cases:
- Get today's date for logging and timestamping
- Set default dates in forms and workflows
- Calculate date-based conditions
- Track daily operations and schedules

**Tags:** date, today, now, current

**Fields:**


## BaseUrl

Extract the base URL from a given URL.

Use cases:
- Get domain name from full URLs
- Clean up URLs for comparison
- Extract root website addresses
- Standardize URL formats

**Tags:** url parsing, domain extraction, web utilities

**Fields:**
- **url**: The URL to extract the base from (str)


## ExtractAudio

Extract audio elements from HTML content.

Use cases:
- Collect audio sources from web pages
- Analyze audio usage on websites
- Create audio playlists

**Tags:** extract, audio, src

**Fields:**
- **html**: The HTML content to extract audio from. (str)
- **base_url**: The base URL of the page, used to resolve relative audio URLs. (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.beautifulsoup.ExtractAudio.OutputType, NoneType]


## ExtractImages

Extract images from HTML content.

Use cases:
- Collect images from web pages
- Analyze image usage on websites
- Create image galleries

**Tags:** extract, images, src

**Fields:**
- **html**: The HTML content to extract images from. (str)
- **base_url**: The base URL of the page, used to resolve relative image URLs. (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.beautifulsoup.ExtractImages.OutputType, NoneType]


## ExtractLinks

Extract all links from HTML content with type classification.

Use cases:
- Analyze website structure and navigation
- Discover related content and resources
- Build sitemaps and link graphs
- Find internal and external references
- Collect URLs for further processing

**Tags:** extract, links, urls, web scraping, html

**Fields:**
- **html**: The HTML content to extract links from. (str)
- **base_url**: The base URL of the page, used to determine internal/external links. (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.beautifulsoup.ExtractLinks.OutputType, NoneType]


## ExtractMetadata

Extract metadata from HTML content.

Use cases:
- Analyze SEO elements
- Gather page information
- Extract structured data

**Tags:** extract, metadata, seo

**Fields:**
- **html**: The HTML content to extract metadata from. (str)


## ExtractVideos

Extract videos from HTML content.

Use cases:
- Collect video sources from web pages
- Analyze video usage on websites
- Create video playlists

**Tags:** extract, videos, src

**Fields:**
- **html**: The HTML content to extract videos from. (str)
- **base_url**: The base URL of the page, used to resolve relative video URLs. (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.beautifulsoup.ExtractVideos.OutputType, NoneType]


## HTMLToText

Converts HTML to plain text by removing tags and decoding entities using BeautifulSoup.

Use cases:
- Cleaning HTML content for text analysis
- Extracting readable content from web pages
- Preparing HTML data for natural language processing

**Tags:** html, text, convert

**Fields:**
- **text** (str)
- **preserve_linebreaks**: Convert block-level elements to newlines (bool)


## WebsiteContentExtractor

Extract main content from a website, removing navigation, ads, and other non-essential elements.

Use cases:
- Clean web content for further analysis
- Extract article text from news websites
- Prepare web content for summarization

**Tags:** scrape, web scraping, content extraction, text analysis

**Fields:**
- **html_content**: The raw HTML content of the website. (str)


## CombineImageGrid

Combine a grid of image tiles into a single image.

Use cases:
- Reassemble processed image chunks
- Create composite images from smaller parts
- Merge tiled image data from distributed processing

**Tags:** image, grid, combine, tiles

**Fields:**
- **tiles**: List of image tiles to combine. (list[nodetool.metadata.types.ImageRef])
- **columns**: Number of columns in the grid. (int)


## SliceImageGrid

Slice an image into a grid of tiles.

Use cases:
- Prepare large images for processing in smaller chunks
- Create image puzzles or mosaic effects
- Distribute image processing tasks across multiple workers

**Tags:** image, grid, slice, tiles

**Fields:**
- **image**: The image to slice into a grid. (ImageRef)
- **columns**: Number of columns in the grid. (int)
- **rows**: Number of rows in the grid. (int)


## BaseGetJSONPath

Base class for extracting typed data from a JSON object using a path expression.

Examples for an object {"a": {"b": {"c": 1}}}
"a.b.c" -> 1
"a.b" -> {"c": 1}
"a" -> {"b": {"c": 1}}

Use cases:
- Navigate complex JSON structures
- Extract specific values from nested JSON with type safety

**Tags:** json, path, extract

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)


## FilterJSON

Filter JSON array based on a key-value condition.

Use cases:
- Filter arrays of objects
- Search JSON data

**Tags:** json, filter, array

**Fields:**
- **array**: Array of JSON objects to filter (list[dict])
- **key**: Key to filter on (str)
- **value**: Value to match (Any)


## GetJSONPathBool

Extract a boolean value from a JSON path

**Tags:** json, path, extract, boolean

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default**: Default value to return if path is not found (bool)


## GetJSONPathDict

Extract a dictionary value from a JSON path

**Tags:** json, path, extract, object

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default**: Default value to return if path is not found (dict)


## GetJSONPathFloat

Extract a float value from a JSON path

**Tags:** json, path, extract, number

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default**: Default value to return if path is not found (float)


## GetJSONPathInt

Extract an integer value from a JSON path

**Tags:** json, path, extract, number

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default**: Default value to return if path is not found (int)


## GetJSONPathList

Extract a list value from a JSON path

**Tags:** json, path, extract, array

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default**: Default value to return if path is not found (list)


## GetJSONPathStr

Extract a string value from a JSON path

**Tags:** json, path, extract, string

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default**: Default value to return if path is not found (str)


## JSONTemplate

Template JSON strings with variable substitution.

Example:
template: '{"name": "$user", "age": $age}'
values: {"user": "John", "age": 30}
result: '{"name": "John", "age": 30}'

Use cases:
- Create dynamic JSON payloads
- Generate JSON with variable data
- Build API request templates

**Tags:** json, template, substitute, variables

**Fields:**
- **template**: JSON template string with $variable placeholders (str)
- **values**: Dictionary of values to substitute into the template (dict[str, typing.Any])


## LoadJSONAssets

Load JSON files from an asset folder.

**Tags:** load, json, file, import

**Fields:**
- **folder**: The asset folder to load the JSON files from. (FolderRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.lib.json.LoadJSONAssets.OutputType, NoneType]


## ParseDict

Parse a JSON string into a Python dictionary.

Use cases:
- Convert JSON API responses to Python dictionaries
- Process JSON configuration files
- Parse object-like JSON data

**Tags:** json, parse, decode, dictionary

**Fields:**
- **json_string**: JSON string to parse into a dictionary (str)


## ParseList

Parse a JSON string into a Python list.

Use cases:
- Convert JSON array responses to Python lists
- Process JSON data collections
- Parse array-like JSON data

**Tags:** json, parse, decode, array, list

**Fields:**
- **json_string**: JSON string to parse into a list (str)


## StringifyJSON

Convert a Python object to a formatted JSON string.

Use cases:
- Prepare data for API requests
- Save data in JSON format
- Format data for storage or transmission
- Create human-readable JSON output

**Tags:** json, stringify, encode, serialize

**Fields:**
- **data**: Data to convert to JSON (Any)
- **indent**: Number of spaces for indentation (int)


## ValidateJSON

Validate JSON data against a schema.

Use cases:
- Ensure API payloads match specifications
- Validate configuration files

**Tags:** json, validate, schema

**Fields:**
- **data**: JSON data to validate (Any)
- **json_schema**: JSON schema for validation (dict)


## DeleteRequest

Remove a resource from a server using an HTTP DELETE request.

Use cases:
- Delete user accounts
- Remove API resources
- Cancel subscriptions
- Clear cache entries

**Tags:** http, delete, request, url

**Fields:**
- **url**: The URL to make the request to. (str)


## DownloadDataframe

Download data from a URL and return as a dataframe.

Use cases:
- Download CSV data and convert to dataframe
- Fetch JSON data and convert to dataframe
- Retrieve tabular data from APIs
- Process data files from URLs

**Tags:** http, get, request, url, dataframe, csv, json, data

**Fields:**
- **url**: The URL to make the request to. (str)
- **file_format**: The format of the data file (csv, json, tsv). (FileFormat)
- **columns**: The columns of the dataframe. (RecordType)
- **encoding**: The encoding of the text file. (str)
- **delimiter**: The delimiter for CSV/TSV files. (str)


## DownloadFiles

Download files from a list of URLs into a local folder.

Use cases:
- Batch download files from multiple URLs
- Create local copies of remote resources
- Archive web content
- Download datasets

**Tags:** download, files, urls, batch

**Fields:**
- **urls**: List of URLs to download. (list[str])
- **output_folder**: Local folder path where files will be saved. (str)
- **max_concurrent_downloads**: Maximum number of concurrent downloads. (int)

### download_file

**Args:**
- **session (ClientSession)**
- **url (str)**

**Returns:** str

### get_request_kwargs

**Args:**

**Returns:** dict[str, typing.Any]


## FetchPage

Fetch a web page using Selenium and return its content.

Use cases:
- Retrieve content from dynamic websites
- Capture JavaScript-rendered content
- Interact with web applications

**Tags:** selenium, fetch, webpage, http

**Fields:**
- **url**: The URL to fetch the page from. (str)
- **wait_time**: Maximum time to wait for page load (in seconds). (int)


## FilterValidURLs

Filter a list of URLs by checking their validity using HEAD requests.

Use cases:
- Clean URL lists by removing broken links
- Verify resource availability
- Validate website URLs before processing

**Tags:** url validation, http, head request

**Fields:**
- **url**: The URL to make the request to. (str)
- **urls**: List of URLs to validate. (list[str])
- **max_concurrent_requests**: Maximum number of concurrent HEAD requests. (int)

### check_url

**Args:**
- **session (ClientSession)**
- **url (str)**

**Returns:** tuple[str, bool]


## GetRequest

Perform an HTTP GET request to retrieve data from a specified URL.

Use cases:
- Fetch web page content
- Retrieve API data
- Download files
- Check website availability

**Tags:** http, get, request, url

**Fields:**
- **url**: The URL to make the request to. (str)


## GetRequestBinary

Perform an HTTP GET request and return raw binary data.

Use cases:
- Download binary files
- Fetch images or media
- Retrieve PDF documents
- Download any non-text content

**Tags:** http, get, request, url, binary, download

**Fields:**
- **url**: The URL to make the request to. (str)


## GetRequestDocument

Perform an HTTP GET request and return a document

Use cases:
- Download PDF documents
- Retrieve Word documents
- Fetch Excel files
- Download any document format

**Tags:** http, get, request, url, document

**Fields:**
- **url**: The URL to make the request to. (str)


## HTTPBaseNode

Base node for HTTP requests.
http, network, request

Use cases:
- Share common fields for HTTP nodes
- Add custom request parameters in subclasses
- Control visibility of specific request types

**Tags:** 

**Fields:**
- **url**: The URL to make the request to. (str)

### get_request_kwargs

**Args:**

**Returns:** dict[str, typing.Any]


## HeadRequest

Retrieve headers from a resource using an HTTP HEAD request.

Use cases:
- Check resource existence
- Get metadata without downloading content
- Verify authentication or permissions

**Tags:** http, head, request, url

**Fields:**
- **url**: The URL to make the request to. (str)


## ImageDownloader

Download images from list of URLs and return a list of ImageRefs.

Use cases:
- Prepare image datasets for machine learning tasks
- Archive images from web pages
- Process and analyze images extracted from websites

**Tags:** image download, web scraping, data processing

**Fields:**
- **images**: List of image URLs to download. (list[str])
- **base_url**: Base URL to prepend to relative image URLs. (str)
- **max_concurrent_downloads**: Maximum number of concurrent image downloads. (int)

### download_image

**Args:**
- **session (ClientSession)**
- **url (str)**
- **context (ProcessingContext)**

**Returns:** tuple[nodetool.metadata.types.ImageRef | None, str | None]


## JSONGetRequest

Perform an HTTP GET request and parse the response as JSON.

Use cases:
- Fetch data from REST APIs
- Retrieve JSON-formatted responses
- Interface with JSON web services

**Tags:** http, get, request, url, json, api

**Fields:**
- **url**: The URL to make the request to. (str)


## JSONPatchRequest

Partially update resources with JSON data using an HTTP PATCH request.

Use cases:
- Partial updates to API resources
- Modify specific fields without full replacement
- Efficient updates for large objects

**Tags:** http, patch, request, url, json, api

**Fields:**
- **url**: The URL to make the request to. (str)
- **data**: The JSON data to send in the PATCH request. (dict)


## JSONPostRequest

Send JSON data to a server using an HTTP POST request.

Use cases:
- Send structured data to REST APIs
- Create resources with JSON payloads
- Interface with modern web services

**Tags:** http, post, request, url, json, api

**Fields:**
- **url**: The URL to make the request to. (str)
- **data**: The JSON data to send in the POST request. (dict)


## JSONPutRequest

Update resources with JSON data using an HTTP PUT request.

Use cases:
- Update existing API resources
- Replace complete objects in REST APIs
- Set configuration with JSON data

**Tags:** http, put, request, url, json, api

**Fields:**
- **url**: The URL to make the request to. (str)
- **data**: The JSON data to send in the PUT request. (dict)


## PostRequest

Send data to a server using an HTTP POST request.

Use cases:
- Submit form data
- Create new resources on an API
- Upload files
- Authenticate users

**Tags:** http, post, request, url, data

**Fields:**
- **url**: The URL to make the request to. (str)
- **data**: The data to send in the POST request. (str)


## PostRequestBinary

Send data using an HTTP POST request and return raw binary data.

Use cases:
- Upload and receive binary files
- Interact with binary APIs
- Process image or media uploads
- Handle binary file transformations

**Tags:** http, post, request, url, data, binary

**Fields:**
- **url**: The URL to make the request to. (str)
- **data**: The data to send in the POST request. Can be string or binary. (str | bytes)


## PutRequest

Update existing resources on a server using an HTTP PUT request.

Use cases:
- Update user profiles
- Modify existing API resources
- Replace file contents
- Set configuration values

**Tags:** http, put, request, url, data

**Fields:**
- **url**: The URL to make the request to. (str)
- **data**: The data to send in the PUT request. (str)


## PaddleOCRNode

Performs Optical Character Recognition (OCR) on images using PaddleOCR.

Use cases:
- Text extraction from images
- Document digitization
- Receipt/invoice processing
- Handwriting recognition

**Tags:** image, text, ocr, document

**Fields:**
- **image**: The image to perform OCR on (ImageRef)
- **language**: Language code for OCR (OCRLanguage)

### initialize

**Args:**
- **context (ProcessingContext)**

### required_inputs

**Args:**


## Browser

Fetches content from a web page using a headless browser.

Use cases:
- Extract content from JavaScript-heavy websites
- Retrieve text content from web pages
- Get metadata from web pages
- Save extracted content to files

**Tags:** browser, web, scraping, content, fetch

**Fields:**
- **url**: URL to navigate to (str)
- **timeout**: Timeout in milliseconds for page navigation (int)

### get_timeout_seconds

Return a conservative overall timeout for the browser task.

Uses the configured page timeout (milliseconds) plus headroom to
cover driver startup and teardown.


**Returns:**

- **float | None**: Timeout in seconds.
**Args:**

**Returns:** float | None


## BrowserNavigation

Navigates and interacts with web pages in a browser session.

Use cases:
- Perform complex web interactions
- Navigate through multi-step web processes
- Extract content after interaction

**Tags:** browser, navigation, interaction, click, extract

**Fields:**
- **url**: URL to navigate to (required for 'goto' action) (str)
- **action**: Navigation or extraction action to perform (Action)
- **selector**: CSS selector for the element to interact with or extract from (str)
- **timeout**: Timeout in milliseconds for the action (int)
- **wait_for**: Optional selector to wait for after performing the action (str)
- **extract_type**: Type of content to extract (for 'extract' action) (ExtractType)
- **attribute**: Attribute name to extract (when extract_type is 'attribute') (str)

### get_timeout_seconds

Return an overall timeout covering navigation and interactions.

Uses the configured action timeout (milliseconds) plus headroom.


**Returns:**

- **float | None**: Timeout in seconds.
**Args:**

**Returns:** float | None


## BrowserUseNode

Browser agent tool that uses browser_use under the hood.
This module provides a tool for running browser-based agents using the browser_use library.
The agent can perform complex web automation tasks like form filling, navigation, data extraction,
and multi-step workflows using natural language instructions.

Use cases:
- Perform complex web automation tasks based on natural language.
- Automate form filling and data entry.
- Scrape data after complex navigation or interaction sequences.
- Automate multi-step web workflows.

**Tags:** 

**Fields:**
- **model**: The model to use for the browser agent. (BrowserUseModel)
- **task**: Natural language description of the browser task to perform. Can include complex multi-step instructions like 'Compare prices between websites', 'Fill out forms', or 'Extract specific data'. (str)
- **timeout**: Maximum time in seconds to allow for task completion. Complex tasks may require longer timeouts. (int)
- **use_remote_browser**: Use a remote browser instead of a local one (bool)


## DownloadFile

Downloads a file from a URL and saves it to disk.

Use cases:
- Download documents, images, or other files from the web
- Save data for further processing
- Retrieve file assets for analysis

**Tags:** download, file, web, save

**Fields:**
- **url**: URL of the file to download (str)


## Screenshot

Takes a screenshot of a web page or specific element.

Use cases:
- Capture visual representation of web pages
- Document specific UI elements
- Create visual records of web content

**Tags:** browser, screenshot, capture, image

**Fields:**
- **url**: URL to navigate to before taking screenshot (str)
- **selector**: Optional CSS selector for capturing a specific element (str)
- **output_file**: Path to save the screenshot (relative to workspace) (str)
- **timeout**: Timeout in milliseconds for page navigation (int)

### get_timeout_seconds

Return a conservative overall timeout for the screenshot task.

Based on navigation timeout (milliseconds) plus headroom.


**Returns:**

- **float | None**: Timeout in seconds.
**Args:**

**Returns:** float | None


## WebFetch

Fetches HTML content from a URL and converts it to text.

Use cases:
- Extract text content from web pages
- Process web content for analysis
- Save web content to files

**Tags:** web, fetch, html, markdown, http

**Fields:**
- **url**: URL to fetch content from (str)
- **selector**: CSS selector to extract specific elements (str)


## ChartRenderer

Node responsible for rendering chart configurations into image format using seaborn.

**Tags:** chart, seaborn, plot, visualization, data

**Fields:**
- **chart_config**: The chart configuration to render. (ChartConfig)
- **width**: The width of the chart in pixels. (int)
- **height**: The height of the chart in pixels. (int)
- **data**: The data to visualize as a pandas DataFrame. (DataframeRef)
- **despine**: Whether to remove top and right spines. (bool)
- **trim_margins**: Whether to use tight layout for margins. (bool)


## AddHeading

Adds a heading to the document

**Tags:** document, docx, heading, format

**Fields:**
- **document**: The document to add the heading to (DocumentRef)
- **text**: The heading text (str)
- **level**: Heading level (1-9) (int)


## AddImage

Adds an image to the document

**Tags:** document, docx, image, format

**Fields:**
- **document**: The document to add the image to (DocumentRef)
- **image**: The image to add (ImageRef)
- **width**: Image width in inches (float)
- **height**: Image height in inches (float)


## AddPageBreak

Adds a page break to the document

**Tags:** document, docx, format, layout

**Fields:**
- **document**: The document to add the page break to (DocumentRef)


## AddParagraph

Adds a paragraph of text to the document

**Tags:** document, docx, text, format

**Fields:**
- **document**: The document to add the paragraph to (DocumentRef)
- **text**: The paragraph text (str)
- **alignment**: Text alignment (ParagraphAlignment)
- **bold**: Make text bold (bool)
- **italic**: Make text italic (bool)
- **font_size**: Font size in points (int)


## AddTable

Adds a table to the document

**Tags:** document, docx, table, format

**Fields:**
- **document**: The document to add the table to (DocumentRef)
- **data**: The data to add to the table (DataframeRef)


## CreateDocument

Creates a new Word document

**Tags:** document, docx, file, create

**Fields:**


## LoadWordDocument

Loads a Word document from disk

**Tags:** document, docx, file, load, input

**Fields:**
- **path**: Path to the document to load (str)


## SaveDocument

Writes the document to a file

**Tags:** document, docx, file, save, output

**Fields:**
- **document**: The document to write (DocumentRef)
- **path**: The folder to write the document to. (FilePath)
- **filename**: 
        The filename to write the document to.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## SetDocumentProperties

Sets document metadata properties

**Tags:** document, docx, metadata, properties

**Fields:**
- **document**: The document to modify (DocumentRef)
- **title**: Document title (str)
- **author**: Document author (str)
- **subject**: Document subject (str)
- **keywords**: Document keywords (str)


## FormatUUID

Format a UUID string in different representations.

Use cases:
- Convert UUID to different formats
- Generate URN representations
- Format UUIDs for specific use cases

**Tags:** uuid, format, convert, hex, urn, identifier

**Fields:**
- **uuid_string**: UUID string to format (str)
- **format**: Output format (standard, hex, urn, int, bytes_hex) (UUIDFormat)


## GenerateUUID1

Generate a time-based UUID (version 1).

Use cases:
- Create sortable unique identifiers
- Generate time-ordered IDs
- Track creation timestamps in IDs

**Tags:** uuid, time, identifier, unique, guid, timestamp

**Fields:**


## GenerateUUID3

Generate a name-based UUID using MD5 (version 3).

Use cases:
- Create deterministic IDs from names
- Generate consistent identifiers for the same input
- Map names to unique identifiers

**Tags:** uuid, name, identifier, unique, guid, md5, deterministic

**Fields:**
- **namespace**: Namespace (dns, url, oid, x500, or a UUID string) (str)
- **name**: Name to generate UUID from (str)


## GenerateUUID4

Generate a random UUID (version 4).

Use cases:
- Create unique identifiers for records
- Generate session IDs
- Produce random unique keys

**Tags:** uuid, random, identifier, unique, guid

**Fields:**


## GenerateUUID5

Generate a name-based UUID using SHA-1 (version 5).

Use cases:
- Create deterministic IDs from names (preferred over UUID3)
- Generate consistent identifiers for the same input
- Map names to unique identifiers with better collision resistance

**Tags:** uuid, name, identifier, unique, guid, sha1, deterministic

**Fields:**
- **namespace**: Namespace (dns, url, oid, x500, or a UUID string) (str)
- **name**: Name to generate UUID from (str)


## IsValidUUID

Check if a string is a valid UUID.

Use cases:
- Validate user input
- Filter valid UUIDs from a dataset
- Conditional workflow based on UUID validity

**Tags:** uuid, validate, check, verify, identifier

**Fields:**
- **uuid_string**: String to check (str)


## ParseUUID

Parse and validate a UUID string.

Use cases:
- Validate UUID format
- Normalize UUID strings
- Extract UUID version information

**Tags:** uuid, parse, validate, check, identifier

**Fields:**
- **uuid_string**: UUID string to parse (str)


## AutoFitColumns

Automatically adjusts column widths to fit content.

Use cases:
- Improve spreadsheet readability
- Professional presentation

**Tags:** excel, format, columns

**Fields:**
- **workbook**: The Excel workbook to format (ExcelRef)
- **sheet_name**: Target worksheet name (str)


## CreateWorkbook

Creates a new Excel workbook.

Use cases:
- Initialize new Excel files
- Start spreadsheet creation workflows

**Tags:** excel, workbook, create

**Fields:**
- **sheet_name**: Name for the first worksheet (str)


## DataFrameToExcel

Writes a DataFrame to an Excel worksheet.

Use cases:
- Export data analysis results
- Create reports from data

**Tags:** excel, dataframe, export

**Fields:**
- **workbook**: The Excel workbook to write to (ExcelRef)
- **dataframe**: DataFrame to write (DataframeRef)
- **sheet_name**: Target worksheet name (str)
- **start_cell**: Starting cell for data (str)
- **include_header**: Include column headers (bool)


## ExcelToDataFrame

Reads an Excel worksheet into a pandas DataFrame.

Use cases:
- Import Excel data for analysis
- Process spreadsheet contents

**Tags:** excel, dataframe, import

**Fields:**
- **workbook**: The Excel workbook to read from (ExcelRef)
- **sheet_name**: Source worksheet name (str)
- **has_header**: First row contains headers (bool)


## FormatCells

Applies formatting to a range of cells.

Use cases:
- Highlight important data
- Create professional looking reports

**Tags:** excel, format, style

**Fields:**
- **workbook**: The Excel workbook to format (ExcelRef)
- **sheet_name**: Target worksheet name (str)
- **cell_range**: Cell range to format (e.g. 'A1:B10') (str)
- **bold**: Make text bold (bool)
- **background_color**: Background color in hex format (e.g. 'FFFF00' for yellow) (str)
- **text_color**: Text color in hex format (str)


## SaveWorkbook

Saves an Excel workbook to disk.

Use cases:
- Export final spreadsheet
- Save work in progress

**Tags:** excel, save, export

**Fields:**
- **workbook**: The Excel workbook to save (ExcelRef)
- **folder**: The folder to save the file to. (FilePath)
- **filename**: 
        The filename to save the file to.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## CDL

ASC CDL (Color Decision List) color correction.

Use cases:
- Apply industry-standard CDL color correction
- Exchange color grades between different software
- Apply precise mathematical color transformations
- Create consistent looks across multiple shots

Formula: output = (input * slope + offset) ^ power
Followed by saturation adjustment.

**Tags:** cdl, slope, offset, power, saturation, asc, color decision list

**Fields:**
- **image**: The image to color correct. (ImageRef)
- **slope_r**: Red slope (multiplier). (float)
- **slope_g**: Green slope (multiplier). (float)
- **slope_b**: Blue slope (multiplier). (float)
- **offset_r**: Red offset (addition). (float)
- **offset_g**: Green offset (addition). (float)
- **offset_b**: Blue offset (addition). (float)
- **power_r**: Red power (gamma). (float)
- **power_g**: Green power (gamma). (float)
- **power_b**: Blue power (gamma). (float)
- **saturation**: Saturation adjustment. (float)


## ColorBalance

Adjust color temperature and tint for white balance correction.

Use cases:
- Correct white balance in photos and video
- Warm up or cool down the overall image
- Fix color casts from mixed lighting
- Create mood through color temperature shifts

**Tags:** white balance, temperature, tint, color balance, warm, cool

**Fields:**
- **image**: The image to adjust. (ImageRef)
- **temperature**: Color temperature. Positive = warmer (orange), negative = cooler (blue). (float)
- **tint**: Color tint. Positive = magenta, negative = green. (float)


## Curves

RGB curves adjustment with control points for precise tonal control.

Use cases:
- Create custom contrast curves
- Adjust specific tonal ranges precisely
- Create cross-processed or stylized looks
- Match the tonal characteristics of film stocks

**Tags:** curves, rgb, tonal, contrast, levels

**Fields:**
- **image**: The image to adjust. (ImageRef)
- **black_point**: Input black point (lifts shadows). (float)
- **white_point**: Input white point (compresses highlights). (float)
- **shadows**: Shadow curve adjustment. (float)
- **midtones**: Midtone curve adjustment (gamma). (float)
- **highlights**: Highlight curve adjustment. (float)
- **red_midtones**: Red channel midtone adjustment. (float)
- **green_midtones**: Green channel midtone adjustment. (float)
- **blue_midtones**: Blue channel midtone adjustment. (float)


## Exposure

Comprehensive tonal exposure controls similar to Lightroom/Camera Raw.

Use cases:
- Correct over/underexposed images
- Recover highlight and shadow detail
- Adjust overall contrast and tonal range
- Fine-tune the brightness of specific tonal regions

**Tags:** exposure, contrast, highlights, shadows, whites, blacks, tonal

**Fields:**
- **image**: The image to adjust. (ImageRef)
- **exposure**: Exposure adjustment in stops. Affects entire image. (float)
- **contrast**: Contrast adjustment. Affects midtone separation. (float)
- **highlights**: Highlight recovery/boost. Affects brightest areas. (float)
- **shadows**: Shadow recovery/darken. Affects darkest areas. (float)
- **whites**: White point adjustment. Sets the brightest white. (float)
- **blacks**: Black point adjustment. Sets the darkest black. (float)


## FilmLook

Apply preset cinematic film looks with adjustable intensity.

Use cases:
- Quickly apply popular cinematic color grades
- Create consistent looks across multiple images
- Emulate classic film stock characteristics
- Starting point for custom color grading

**Tags:** film look, cinematic, preset, movie, lut, color grade

**Fields:**
- **image**: The image to apply the film look to. (ImageRef)
- **preset**: The cinematic look to apply. (FilmLookPreset)
- **intensity**: Intensity of the effect. 0=none, 1=full, 2=exaggerated. (float)


## HSLAdjust

Adjust hue, saturation, and luminance for specific color ranges.

Use cases:
- Shift specific colors (e.g., make blues more cyan)
- Desaturate or boost individual color ranges
- Brighten or darken specific colors
- Create color-specific looks (teal skies, orange skin)

**Tags:** hsl, hue, saturation, luminance, selective color, color range

**Fields:**
- **image**: The image to adjust. (ImageRef)
- **color_range**: The color range to adjust. (ColorRange)
- **hue_shift**: Hue shift for the selected color range. -1 to 1 = -180 to +180 degrees. (float)
- **saturation**: Saturation adjustment for the selected color range. (float)
- **luminance**: Luminance adjustment for the selected color range. (float)


## LiftGammaGain

Three-way color corrector for shadows, midtones, and highlights.

Use cases:
- Apply the industry-standard three-way color correction
- Balance colors across different tonal ranges
- Create color contrast between shadows and highlights
- Match footage from different sources

Lift affects shadows, Gamma affects midtones, Gain affects highlights.
Each control adjusts both luminance and color for its tonal range.

**Tags:** lift, gamma, gain, color wheels, primary correction, shadows, midtones, highlights

**Fields:**
- **image**: The image to color correct. (ImageRef)
- **lift_r**: Red lift (shadow color shift). (float)
- **lift_g**: Green lift (shadow color shift). (float)
- **lift_b**: Blue lift (shadow color shift). (float)
- **lift_master**: Master lift (shadow brightness). (float)
- **gamma_r**: Red gamma (midtone adjustment). (float)
- **gamma_g**: Green gamma (midtone adjustment). (float)
- **gamma_b**: Blue gamma (midtone adjustment). (float)
- **gamma_master**: Master gamma (overall midtones). (float)
- **gain_r**: Red gain (highlight multiplier). (float)
- **gain_g**: Green gain (highlight multiplier). (float)
- **gain_b**: Blue gain (highlight multiplier). (float)
- **gain_master**: Master gain (overall brightness). (float)


## SaturationVibrance

Adjust color saturation with vibrance protection for skin tones.

Use cases:
- Boost color intensity without clipping
- Protect skin tones while increasing saturation
- Create desaturated or oversaturated looks
- Fine-tune color intensity independently

**Tags:** saturation, vibrance, color intensity, skin tones

**Fields:**
- **image**: The image to adjust. (ImageRef)
- **saturation**: Global saturation. 0 = no change, -1 = grayscale, 1 = 2x saturation. (float)
- **vibrance**: Smart saturation that protects already-saturated colors and skin tones. (float)


## SplitToning

Apply different color tints to shadows and highlights.

Use cases:
- Create classic teal and orange looks
- Add color contrast between shadows and highlights
- Emulate film processing techniques
- Create stylized color-graded images

**Tags:** split toning, shadows, highlights, tint, duotone

**Fields:**
- **image**: The image to apply split toning to. (ImageRef)
- **shadow_hue**: Hue of shadow tint in degrees (0=red, 120=green, 240=blue). (float)
- **shadow_saturation**: Saturation of shadow tint. (float)
- **highlight_hue**: Hue of highlight tint in degrees. (float)
- **highlight_saturation**: Saturation of highlight tint. (float)
- **balance**: Balance between shadows (-1) and highlights (+1). (float)


## Vignette

Apply cinematic vignette effect to darken or lighten image edges.

Use cases:
- Draw attention to the center of the image
- Create a classic cinematic look
- Simulate lens light falloff
- Add subtle framing to photos

**Tags:** vignette, edge, darken, focus, cinematic

**Fields:**
- **image**: The image to apply vignette to. (ImageRef)
- **amount**: Vignette amount. Positive darkens edges, negative lightens. (float)
- **midpoint**: Distance from center where vignette begins (0=center, 1=edges). (float)
- **roundness**: Shape of vignette. 0=oval matching image aspect, 1=circular, -1=rectangular. (float)
- **feather**: Softness of the vignette edge. (float)


## Blur

Apply a Gaussian blur effect to an image.

- Soften images or reduce noise and detail
- Make focal areas stand out by blurring surroundings
- Protect privacy by blurring sensitive information

**Tags:** image, filter, blur

**Fields:**
- **image**: The image to blur. (ImageRef)
- **radius**: Blur radius. (int)


## Canny

Apply Canny edge detection to an image.

- Highlight areas of rapid intensity change
- Outline object boundaries and structure
- Enhance inputs for object detection and image segmentation

**Tags:** image, filter, edges

**Fields:**
- **image**: The image to canny. (ImageRef)
- **low_threshold**: Low threshold. (int)
- **high_threshold**: High threshold. (int)


## Contour

Apply a contour filter to highlight image edges.

- Extract key features from complex images
- Aid pattern recognition and object detection
- Create stylized contour sketch art effects

**Tags:** image, filter, contour

**Fields:**
- **image**: The image to contour. (ImageRef)


## ConvertToGrayscale

Convert an image to grayscale.

- Simplify images for feature and edge detection
- Prepare images for shape-based machine learning
- Create vintage or monochrome aesthetic effects

**Tags:** image, grayscale

**Fields:**
- **image**: The image to convert. (ImageRef)


## Emboss

Apply an emboss filter for a 3D raised effect.

- Add texture and depth to photos
- Create visually interesting graphics
- Incorporate unique effects in digital artwork

**Tags:** image, filter, emboss

**Fields:**
- **image**: The image to emboss. (ImageRef)


## Expand

Add a border around an image to increase its size.

- Make images stand out by adding a colored border
- Create framed photo effects
- Separate image content from surroundings

**Tags:** image, border, expand

**Fields:**
- **image**: The image to expand. (ImageRef)
- **border**: Border size. (int)
- **fill**: Fill color. (int)


## FindEdges

Detect and highlight edges in an image.

- Analyze structural patterns in images
- Aid object detection in computer vision
- Detect important features like corners and ridges

**Tags:** image, filter, edges

**Fields:**
- **image**: The image to find edges. (ImageRef)


## GetChannel

Extract a specific color channel from an image.

- Isolate color information for image analysis
- Manipulate specific color components in graphic design
- Enhance or reduce visibility of certain colors

**Tags:** image, color, channel, isolate, extract

**Fields:**
- **image**: The image to get the channel from. (ImageRef)
- **channel** (ChannelEnum)


## Invert

Invert the colors of an image.

- Create negative versions of images for visual effects
- Analyze image data by bringing out hidden details
- Preprocess images for operations that work better on inverted colors

**Tags:** image, filter, invert

**Fields:**
- **image**: The image to adjust the brightness for. (ImageRef)


## Posterize

Reduce the number of colors in an image for a poster-like effect.

- Create graphic art by simplifying image colors
- Apply artistic effects to photographs
- Generate visually compelling content for advertising

**Tags:** image, filter, posterize

**Fields:**
- **image**: The image to posterize. (ImageRef)
- **bits**: Number of bits to posterize to. (int)


## Smooth

Apply smoothing to reduce image noise and detail.

- Enhance visual aesthetics of images
- Improve object detection by reducing irrelevant details
- Aid facial recognition by simplifying images

**Tags:** image, filter, smooth

**Fields:**
- **image**: The image to smooth. (ImageRef)


## Solarize

Apply a solarize effect to partially invert image tones.

- Create surreal artistic photo effects
- Enhance visual data by making certain elements more prominent
- Add a unique style to images for graphic design

**Tags:** image, filter, solarize

**Fields:**
- **image**: The image to solarize. (ImageRef)
- **threshold**: Threshold for solarization. (int)


## Background

The Background Node creates a blank background.
This node is mainly used for generating a base layer for image processing tasks. It produces a uniform image, having a user-specified width, height and color. The color is given in a hexadecimal format, defaulting to white if not specified.

#### Applications
- As a base layer for creating composite images.
- As a starting point for generating patterns or graphics.
- When blank backgrounds of specific colors are required for visualization tasks.

**Tags:** image, background, blank, base, layer

**Fields:**
- **width** (int)
- **height** (int)
- **color** (ColorRef)


## GaussianNoise

This node creates and adds Gaussian noise to an image.

The Gaussian Noise Node is designed to simulate realistic distortions that can occur in a photographic image. It generates a noise-filled image using the Gaussian (normal) distribution. The noise level can be adjusted using the mean and standard deviation parameters.

#### Applications
- Simulating sensor noise in synthetic data.
- Testing image-processing algorithms' resilience to noise.
- Creating artistic effects in images.

**Tags:** image, noise, gaussian, distortion, artifact

**Fields:**
- **mean** (float)
- **stddev** (float)
- **width** (int)
- **height** (int)


## RenderText

This node allows you to add text to images.
This node takes text, font updates, coordinates (where to place the text), and an image to work with. A user can use the Render Text Node to add a label or title to an image, watermark an image, or place a caption directly on an image.

The Render Text Node offers customizable options, including the ability to choose the text's font, size, color, and alignment (left, center, or right). Text placement can also be defined, providing flexibility to place the text wherever you see fit.

#### Applications
- Labeling images in a image gallery or database.
- Watermarking images for copyright protection.
- Adding custom captions to photographs.
- Creating instructional images to guide the reader's view.

**Tags:** text, font, label, title, watermark, caption, image, overlay

**Fields:**
- **text**: The text to render. (str)
- **font**: The font to use. (FontRef)
- **x**: The x coordinate. (int)
- **y**: The y coordinate. (int)
- **size**: The font size. (int)
- **color**: The font color. (ColorRef)
- **align** (TextAlignment)
- **image**: The image to render on. (ImageRef)


## Blend

Blend two images with adjustable alpha mixing.

Use cases:
- Create smooth transitions between images
- Adjust opacity of overlays
- Combine multiple exposures or effects

**Tags:** blend, mix, fade, transition

**Fields:**
- **image1**: The first image to blend. (ImageRef)
- **image2**: The second image to blend. (ImageRef)
- **alpha**: The mix ratio. (float)


## Composite

Combine two images using a mask for advanced compositing.

Use cases:
- Create complex image compositions
- Apply selective blending or effects
- Implement advanced photo editing techniques

**Tags:** composite, mask, blend, layering

**Fields:**
- **image1**: The first image to composite. (ImageRef)
- **image2**: The second image to composite. (ImageRef)
- **mask**: The mask to composite with. (ImageRef)


## AdaptiveContrast

Applies localized contrast enhancement using adaptive techniques.

Use cases:
- Improve visibility in images with varying lighting conditions
- Prepare images for improved feature detection in computer vision

**Tags:** image, contrast, enhance

**Fields:**
- **image**: The image to adjust the contrast for. (ImageRef)
- **clip_limit**: Clip limit for adaptive contrast. (float)
- **grid_size**: Grid size for adaptive contrast. (int)


## AutoContrast

Automatically adjusts image contrast for enhanced visual quality.

Use cases:
- Enhance image clarity for better visual perception
- Pre-process images for computer vision tasks
- Improve photo aesthetics in editing workflows

**Tags:** image, contrast, balance

**Fields:**
- **image**: The image to adjust the contrast for. (ImageRef)
- **cutoff**: Represents the percentage of pixels to ignore at both the darkest and lightest ends of the histogram. A cutoff value of 5 means ignoring the darkest 5% and the lightest 5% of pixels, enhancing overall contrast by stretching the remaining pixel values across the full brightness range. (int)


## Brightness

Adjusts overall image brightness to lighten or darken.

Use cases:
- Correct underexposed or overexposed photographs
- Enhance visibility of dark image regions
- Prepare images for consistent display across devices

**Tags:** image, brightness, enhance

**Fields:**
- **image**: The image to adjust the brightness for. (ImageRef)
- **factor**: Factor to adjust the brightness. 1.0 means no change. (float | int)


## Color

Adjusts color intensity of an image.

Use cases:
- Enhance color vibrancy in photographs
- Correct color imbalances in digital images
- Prepare images for consistent brand color representation

**Tags:** image, color, enhance

**Fields:**
- **image**: The image to adjust the brightness for. (ImageRef)
- **factor**: Factor to adjust the contrast. 1.0 means no change. (float)


## Contrast

Adjusts image contrast to modify light-dark differences.

Use cases:
- Enhance visibility of details in low-contrast images
- Prepare images for visual analysis or recognition tasks
- Create dramatic effects in artistic photography

**Tags:** image, contrast, enhance

**Fields:**
- **image**: The image to adjust the brightness for. (ImageRef)
- **factor**: Factor to adjust the contrast. 1.0 means no change. (float)


## Detail

Enhances fine details in images.

Use cases:
- Improve clarity of textural elements in photographs
- Enhance visibility of small features for analysis
- Prepare images for high-resolution display or printing

**Tags:** image, detail, enhance

**Fields:**
- **image**: The image to detail. (ImageRef)


## EdgeEnhance

Enhances edge visibility by increasing contrast along boundaries.

Use cases:
- Improve object boundary detection for computer vision
- Highlight structural elements in technical drawings
- Prepare images for feature extraction in image analysis

**Tags:** image, edge, enhance

**Fields:**
- **image**: The image to edge enhance. (ImageRef)


## Equalize

Enhances image contrast by equalizing intensity distribution.

Use cases:
- Improve visibility in poorly lit images
- Enhance details for image analysis tasks
- Normalize image data for machine learning

**Tags:** image, contrast, histogram

**Fields:**
- **image**: The image to equalize. (ImageRef)


## RankFilter

Applies rank-based filtering to enhance or smooth image features.

Use cases:
- Reduce noise while preserving edges in images
- Enhance specific image features based on local intensity
- Pre-process images for improved segmentation results

**Tags:** image, filter, enhance

**Fields:**
- **image**: The image to rank filter. (ImageRef)
- **size**: Rank filter size. (int)
- **rank**: Rank filter rank. (int)


## Sharpen

Enhances image detail by intensifying local pixel contrast.

Use cases:
- Improve clarity of photographs for print or display
- Refine texture details in product photography
- Enhance readability of text in document images

**Tags:** image, sharpen, clarity

**Fields:**
- **image**: The image to sharpen. (ImageRef)


## Sharpness

Adjusts image sharpness to enhance or reduce detail clarity.

Use cases:
- Enhance photo details for improved visual appeal
- Refine images for object detection tasks
- Correct slightly blurred images

**Tags:** image, clarity, sharpness

**Fields:**
- **image**: The image to adjust the brightness for. (ImageRef)
- **factor**: Factor to adjust the contrast. 1.0 means no change. (float)


## UnsharpMask

Sharpens images using the unsharp mask technique.

Use cases:
- Enhance edge definition in photographs
- Improve perceived sharpness of digital artwork
- Prepare images for high-quality printing or display

**Tags:** image, sharpen, enhance

**Fields:**
- **image**: The image to unsharp mask. (ImageRef)
- **radius**: Unsharp mask radius. (int)
- **percent**: Unsharp mask percent. (int)
- **threshold**: Unsharp mask threshold. (int)


## Reshape1D

Reshape an array to a 1D shape without changing its data.

Use cases:
- Flatten multi-dimensional data for certain algorithms
- Convert images to vector form for machine learning
- Prepare data for 1D operations

**Tags:** array, reshape, vector, flatten

**Fields:**
- **values**: The input array to reshape (NPArray)
- **num_elements**: The number of elements (int)


## Reshape2D

Reshape an array to a new shape without changing its data.

Use cases:
- Convert between different dimensional representations
- Prepare data for specific model architectures
- Flatten or unflatten arrays

**Tags:** array, reshape, dimensions, structure

**Fields:**
- **values**: The input array to reshape (NPArray)
- **num_rows**: The number of rows (int)
- **num_cols**: The number of columns (int)


## Reshape3D

Reshape an array to a 3D shape without changing its data.

Use cases:
- Convert data for 3D visualization
- Prepare image data with channels
- Structure data for 3D convolutions

**Tags:** array, reshape, dimensions, volume

**Fields:**
- **values**: The input array to reshape (NPArray)
- **num_rows**: The number of rows (int)
- **num_cols**: The number of columns (int)
- **num_depths**: The number of depths (int)


## Reshape4D

Reshape an array to a 4D shape without changing its data.

Use cases:
- Prepare batch data for neural networks
- Structure spatiotemporal data
- Format data for 3D image processing with channels

**Tags:** array, reshape, dimensions, batch

**Fields:**
- **values**: The input array to reshape (NPArray)
- **num_rows**: The number of rows (int)
- **num_cols**: The number of columns (int)
- **num_depths**: The number of depths (int)
- **num_channels**: The number of channels (int)


## BinaryOperation

**Fields:**
- **a** (int | float | nodetool.metadata.types.NPArray)
- **b** (int | float | nodetool.metadata.types.NPArray)

### operation

**Args:**
- **a (ndarray)**
- **b (ndarray)**

**Returns:** ndarray


## ArgMaxArray

Find indices of maximum values along a specified axis of a array.

Use cases:
- Determine winning classes in classification tasks
- Find peaks in signal processing
- Locate best-performing items in datasets

**Tags:** array, argmax, index, maximum

**Fields:**
- **values**: Input array (NPArray)
- **axis**: Axis along which to find maximum indices (int)


## ArgMinArray

Find indices of minimum values along a specified axis of a array.

Use cases:
- Locate lowest-performing items in datasets
- Find troughs in signal processing
- Determine least likely classes in classification tasks

**Tags:** array, argmin, index, minimum

**Fields:**
- **values**: Input array (NPArray)
- **axis**: Axis along which to find minimum indices (int)


## MaxArray

Compute the maximum value along a specified axis of a array.

Use cases:
- Find peak values in time series data
- Implement max pooling in neural networks
- Determine highest scores across multiple categories

**Tags:** array, maximum, reduction, statistics

**Fields:**
- **values**: Input array (NPArray)
- **axis**: Axis along which to compute maximum (int)


## MeanArray

Compute the mean value along a specified axis of a array.

Use cases:
- Calculate average values in datasets
- Implement mean pooling in neural networks
- Compute centroids in clustering algorithms

**Tags:** array, average, reduction, statistics

**Fields:**
- **values**: Input array (NPArray)
- **axis**: Axis along which to compute mean (int)


## MinArray

Calculate the minimum value along a specified axis of a array.

Use cases:
- Find lowest values in datasets
- Implement min pooling in neural networks
- Determine minimum thresholds across categories

**Tags:** array, minimum, reduction, statistics

**Fields:**
- **values**: Input array (NPArray)
- **axis**: Axis along which to compute minimum (int)


## SumArray

Calculate the sum of values along a specified axis of a array.

Use cases:
- Compute total values across categories
- Implement sum pooling in neural networks
- Calculate cumulative metrics in time series data

**Tags:** array, summation, reduction, statistics

**Fields:**
- **values**: Input array (NPArray)
- **axis**: Axis along which to compute sum (int)


## ArrayToList

Convert a array to a nested list structure.

Use cases:
- Prepare array data for JSON serialization
- Convert array outputs to Python data structures
- Interface array data with non-array operations

**Tags:** array, list, conversion, type

**Fields:**
- **values**: Array to convert to list (NPArray)


## ArrayToScalar

Convert a single-element array to a scalar value.

Use cases:
- Extract final results from array computations
- Prepare values for non-array operations
- Simplify output for human-readable results

**Tags:** array, scalar, conversion, type

**Fields:**
- **values**: Array to convert to scalar (NPArray)


## ConvertToArray

Convert PIL Image to normalized tensor representation.

Use cases:
- Prepare images for machine learning models
- Convert between image formats for processing
- Normalize image data for consistent calculations

**Tags:** image, tensor, conversion, normalization

**Fields:**
- **image**: The input image to convert to a tensor. The image should have either 1 (grayscale), 3 (RGB), or 4 (RGBA) channels. (ImageRef)


## ConvertToAudio

Converts a array object back to an audio file.

Use cases:
- Save processed audio data as a playable file
- Convert generated or modified audio arrays to audio format
- Output results of audio processing pipelinesr

**Tags:** audio, conversion, array

**Fields:**
- **values**: The array to convert to an audio file. (NPArray)
- **sample_rate**: The sample rate of the audio file. (int)


## ConvertToImage

Convert array data to PIL Image format.

Use cases:
- Visualize array data as images
- Save processed array results as images
- Convert model outputs back to viewable format

**Tags:** array, image, conversion, denormalization

**Fields:**
- **values**: The input array to convert to an image. Should have either 1, 3, or 4 channels. (NPArray)


## ListToArray

Convert a list of values to a array.

Use cases:
- Prepare list data for array operations
- Create arrays from Python data structures
- Convert sequence data to array format

**Tags:** list, array, conversion, type

**Fields:**
- **values**: List of values to convert to array (list[typing.Any])


## ScalarToArray

Convert a scalar value to a single-element array.

Use cases:
- Prepare scalar inputs for array operations
- Create constant arrays for computations
- Initialize array values in workflows

**Tags:** scalar, array, conversion, type

**Fields:**
- **value**: Scalar value to convert to array (float | int)


## AbsArray

Compute the absolute value of each element in a array.

Use cases:
- Calculate magnitudes of complex numbers
- Preprocess data for certain algorithms
- Implement activation functions in neural networks

**Tags:** array, absolute, magnitude

**Fields:**
- **values**: The input array to compute the absolute values from. (NPArray)


## CosineArray

Computes the cosine of input angles in radians.

Use cases:
- Calculating horizontal components in physics
- Creating circular motions
- Phase calculations in signal processing

**Tags:** math, trigonometry, cosine, cos

**Fields:**
- **angle_rad** (float | int | nodetool.metadata.types.NPArray)


## ExpArray

Calculate the exponential of each element in a array.

Use cases:
- Implement exponential activation functions
- Calculate growth rates in scientific models
- Transform data for certain statistical analyses

**Tags:** array, exponential, math, activation

**Fields:**
- **values**: Input array (NPArray)


## LogArray

Calculate the natural logarithm of each element in a array.

Use cases:
- Implement log transformations on data
- Calculate entropy in information theory
- Normalize data with large ranges

**Tags:** array, logarithm, math, transformation

**Fields:**
- **values**: Input array (NPArray)


## PowerArray

Raises the base array to the power of the exponent element-wise.

Use cases:
- Calculating compound interest
- Implementing polynomial functions
- Applying non-linear transformations to data

**Tags:** math, exponentiation, power, pow, **

**Fields:**
- **base** (float | int | nodetool.metadata.types.NPArray)
- **exponent** (float | int | nodetool.metadata.types.NPArray)


## SineArray

Computes the sine of input angles in radians.

Use cases:
- Calculating vertical components in physics
- Generating smooth periodic functions
- Audio signal processing

**Tags:** math, trigonometry, sine, sin

**Fields:**
- **angle_rad** (float | int | nodetool.metadata.types.NPArray)


## SqrtArray

Calculates the square root of the input array element-wise.

Use cases:
- Normalizing data
- Calculating distances in Euclidean space
- Finding intermediate values in binary search

**Tags:** math, square root, sqrt, √

**Fields:**
- **values**: Input array (NPArray)


## IndexArray

Select specific indices from an array along a specified axis.

Use cases:
- Extract specific samples from a dataset
- Select particular features or dimensions
- Implement batch sampling operations

**Tags:** array, index, select, subset

**Fields:**
- **values**: The input array to index (NPArray)
- **indices**: The comma separated indices to select (str)
- **axis**: Axis along which to index (int)


## MatMul

Perform matrix multiplication on two input arrays.

Use cases:
- Implement linear transformations
- Calculate dot products of vectors
- Perform matrix operations in neural networks

**Tags:** array, matrix, multiplication, linear algebra

**Fields:**
- **a**: First input array (NPArray)
- **b**: Second input array (NPArray)


## SliceArray

Extract a slice of an array along a specified axis.

Use cases:
- Extract specific time periods from time series data
- Select subset of features from datasets
- Create sliding windows over sequential data

**Tags:** array, slice, subset, index

**Fields:**
- **values**: The input array to slice (NPArray)
- **start**: Starting index (inclusive) (int)
- **stop**: Ending index (exclusive) (int)
- **step**: Step size between elements (int)
- **axis**: Axis along which to slice (int)


## SplitArray

Split an array into multiple sub-arrays along a specified axis.

Use cases:
- Divide datasets into training/validation splits
- Create batches from large arrays
- Separate multi-channel data

**Tags:** array, split, divide, partition

**Fields:**
- **values**: The input array to split (NPArray)
- **num_splits**: Number of equal splits to create (int)
- **axis**: Axis along which to split (int)


## Stack

Stack multiple arrays along a specified axis.

Use cases:
- Combine multiple 2D arrays into a 3D array
- Stack time series data from multiple sources
- Merge feature vectors for machine learning models

**Tags:** array, stack, concatenate, join, merge, axis

**Fields:**
- **arrays**: Arrays to stack (list[nodetool.metadata.types.NPArray])
- **axis**: The axis to stack along. (int)


## TransposeArray

Transpose the dimensions of the input array.

Use cases:
- Convert row vectors to column vectors
- Rearrange data for compatibility with other operations
- Implement certain linear algebra operations

**Tags:** array, transpose, reshape, dimensions

**Fields:**
- **values**: Array to transpose (NPArray)


## PlotArray

Create a plot visualization of array data.

Use cases:
- Visualize trends in array data
- Create charts for reports or dashboards
- Debug array outputs in workflows

**Tags:** array, plot, visualization, graph

**Fields:**
- **values**: Array to plot (NPArray)
- **plot_type**: Type of plot to create (PlotType)


## AddArray

Performs addition on two arrays.

**Tags:** math, plus, add, addition, sum, +

**Fields:**
- **a** (int | float | nodetool.metadata.types.NPArray)
- **b** (int | float | nodetool.metadata.types.NPArray)

### operation

**Args:**
- **a (ndarray)**
- **b (ndarray)**

**Returns:** ndarray


## DivideArray

Divides the first array by the second.

**Tags:** math, division, arithmetic, quotient, /

**Fields:**
- **a** (int | float | nodetool.metadata.types.NPArray)
- **b** (int | float | nodetool.metadata.types.NPArray)

### operation

**Args:**
- **a (ndarray)**
- **b (ndarray)**

**Returns:** ndarray


## ModulusArray

Calculates the element-wise remainder of division.

Use cases:
- Implementing cyclic behaviors
- Checking for even/odd numbers
- Limiting values to a specific range

**Tags:** math, modulo, remainder, mod, %

**Fields:**
- **a** (int | float | nodetool.metadata.types.NPArray)
- **b** (int | float | nodetool.metadata.types.NPArray)

### operation

**Args:**
- **a (ndarray)**
- **b (ndarray)**

**Returns:** ndarray


## MultiplyArray

Multiplies two arrays.

**Tags:** math, product, times, *

**Fields:**
- **a** (int | float | nodetool.metadata.types.NPArray)
- **b** (int | float | nodetool.metadata.types.NPArray)

### operation

**Args:**
- **a (ndarray)**
- **b (ndarray)**

**Returns:** ndarray


## SubtractArray

Subtracts the second array from the first.

**Tags:** math, minus, difference, -

**Fields:**
- **a** (int | float | nodetool.metadata.types.NPArray)
- **b** (int | float | nodetool.metadata.types.NPArray)

### operation

**Args:**
- **a (ndarray)**
- **b (ndarray)**

**Returns:** ndarray


## SaveArray

Save a numpy array to a file in the specified folder.

Use cases:
- Store processed arrays for later use
- Save analysis results
- Create checkpoints in processing pipelines

**Tags:** array, save, file, storage

**Fields:**
- **values**: The array to save. (NPArray)
- **folder**: The folder to save the array in. (FolderRef)
- **name**: 
        The name of the asset to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**


## ChromaNode

Base class for vector database nodes.
vector, base, database, chroma, faiss

Use cases:
- Provide shared helpers for vector indexing and queries
- Disable caching for subclasses
- Convert result IDs into asset references

**Tags:** 

**Fields:**

### load_results

**Args:**
- **context (ProcessingContext)**
- **ids (list[str])**

**Returns:** list[nodetool.metadata.types.AssetRef]


## CollectionNode

Get or create a named vector database collection for storing embeddings.

Use cases:
- Initialize collections for semantic search
- Organize embeddings by topic or domain
- Set up vector stores for RAG applications
- Manage multiple vector databases
- Configure embedding models per collection

**Tags:** vector, embedding, collection, RAG, get, create, chroma

**Fields:**
- **name**: The name of the collection to create (str)
- **embedding_model**: Model to use for embedding, search for nomic-embed-text and download it (LlamaModel)


## Count

Count the number of documents in a collection.

**Tags:** vector, embedding, collection, RAG, chroma

**Fields:**
- **collection**: The collection to count (Collection)


## GetDocuments

Get documents from a chroma collection.

**Tags:** vector, embedding, collection, RAG, retrieve, chroma

**Fields:**
- **collection**: The collection to get (Collection)
- **ids**: The ids of the documents to get (list[str])
- **limit**: The limit of the documents to get (int)
- **offset**: The offset of the documents to get (int)


## HybridSearch

Hybrid search combining semantic and keyword-based search for better retrieval. Uses reciprocal rank fusion to combine results from both methods.

**Tags:** vector, RAG, query, semantic, text, similarity, chroma

**Fields:**
- **collection**: The collection to query (Collection)
- **text**: The text to query (str)
- **n_results**: The number of final results to return (int)
- **k_constant**: Constant for reciprocal rank fusion (default: 60.0) (float)
- **min_keyword_length**: Minimum length for keyword tokens (int)


## IndexAggregatedText

Index multiple text chunks at once with aggregated embeddings from Ollama.

**Tags:** vector, embedding, collection, RAG, index, text, chunk, batch, ollama, chroma

**Fields:**
- **collection**: The collection to index (Collection)
- **document**: The document to index (str)
- **document_id**: The document ID to associate with the text (str)
- **metadata**: The metadata to associate with the text (dict)
- **text_chunks**: List of text chunks to index (list[nodetool.metadata.types.TextChunk | str])
- **context_window**: The context window size to use for the model (int)
- **aggregation**: The aggregation method to use for the embeddings. (EmbeddingAggregation)


## IndexEmbedding

Index a single embedding vector into a Chroma collection with optional metadata. Creates a searchable entry that can be queried for similarity matching.

**Tags:** vector, index, embedding, chroma, storage, RAG

**Fields:**
- **collection**: The collection to index (Collection)
- **embedding**: The embedding to index (NPArray)
- **index_id**: The ID to associate with the embedding (str)
- **metadata**: The metadata to associate with the embedding (dict)


## IndexImage

Index a list of image assets or files.

**Tags:** vector, embedding, collection, RAG, index, image, batch, chroma

**Fields:**
- **collection**: The collection to index (Collection)
- **image**: List of image assets to index (ImageRef)
- **index_id**: The ID to associate with the image, defaults to the URI of the image (str)
- **metadata**: The metadata to associate with the image (dict)
- **upsert**: Whether to upsert the images (bool)


## IndexString

Index a string with a Document ID to a collection.

Use cases:
- Index documents for a vector search

**Tags:** vector, embedding, collection, RAG, index, text, string, chroma

**Fields:**
- **collection**: The collection to index (Collection)
- **text**: Text content to index (str)
- **document_id**: Document ID to associate with the text content (str)
- **metadata**: The metadata to associate with the text (dict)


## IndexTextChunk

Index a single text chunk.

**Tags:** vector, embedding, collection, RAG, index, text, chunk, chroma

**Fields:**
- **collection**: The collection to index (Collection)
- **document_id**: The document ID to associate with the text chunk (str)
- **text**: The text to index (str)
- **metadata**: The metadata to associate with the text chunk (dict)


## Peek

Peek at the documents in a collection.

**Tags:** vector, embedding, collection, RAG, preview, chroma

**Fields:**
- **collection**: The collection to peek (Collection)
- **limit**: The limit of the documents to peek (int)


## QueryImage

Query the index for similar images.

**Tags:** vector, RAG, query, image, search, similarity, chroma

**Fields:**
- **collection**: The collection to query (Collection)
- **image**: The image to query (ImageRef)
- **n_results**: The number of results to return (int)


## QueryText

Query the index for similar text.

**Tags:** vector, RAG, query, text, search, similarity, chroma

**Fields:**
- **collection**: The collection to query (Collection)
- **text**: The text to query (str)
- **n_results**: The number of results to return (int)


## RemoveOverlap

Removes overlapping words between consecutive strings in a list. Splits text into words and matches word sequences for more accurate overlap detection.

**Tags:** vector, RAG, query, text, processing, overlap, deduplication

**Fields:**
- **documents**: List of strings to process for overlap removal (list[str])
- **min_overlap_words**: Minimum number of words that must overlap to be considered (int)


## AddVectors

Add vectors to a FAISS index.

**Tags:** faiss, add, vectors

**Fields:**
- **index**: FAISS index (FaissIndex)
- **vectors**: Vectors to add (n, d) (NPArray)


## AddWithIds

Add vectors with explicit integer IDs to a FAISS index.

**Tags:** faiss, add, ids, vectors

**Fields:**
- **index**: FAISS index (FaissIndex)
- **vectors**: Vectors to add (n, d) (NPArray)
- **ids**: 1-D int64 IDs (n,) (NPArray)


## CreateIndexFlatIP

Create a FAISS IndexFlatIP (inner product / cosine with normalized vectors).

**Tags:** faiss, index, ip, create

**Fields:**
- **dim**: Embedding dimensionality (int)


## CreateIndexFlatL2

Create a FAISS IndexFlatL2.

**Tags:** faiss, index, l2, create

**Fields:**
- **dim**: Embedding dimensionality (int)


## CreateIndexIVFFlat

Create a FAISS IndexIVFFlat (inverted file index with flat quantizer).

**Tags:** faiss, index, ivf, create

**Fields:**
- **dim**: Embedding dimensionality (int)
- **nlist**: Number of Voronoi cells (int)
- **metric**: Distance metric (Metric)


## FaissNode

Base class for FAISS nodes.
vector, faiss, index, search

**Tags:** 

**Fields:**


## Search

Search a FAISS index with query vectors, returning distances and indices.

**Tags:** faiss, search, query, knn

**Fields:**
- **index**: FAISS index (FaissIndex)
- **query**: Query vectors (m, d) or (d,) (NPArray)
- **k**: Number of nearest neighbors (int)
- **nprobe**: nprobe for IVF indices (int)


## TrainIndex

Train a FAISS index with training vectors (required for IVF indices).

**Tags:** faiss, train, index

**Fields:**
- **index**: FAISS index (FaissIndex)
- **vectors**: Training vectors (n, d) (NPArray)


## TextToSpeech

Converts text to speech using OpenAI TTS models.

Use cases:
- Generate spoken content for videos or podcasts
- Create voice-overs for presentations
- Assist visually impaired users with text reading
- Produce audio versions of written content

**Tags:** audio, tts, text-to-speech, voice, synthesis

**Fields:**
- **model** (TtsModel)
- **voice** (Voice)
- **input** (str)
- **speed** (float)


## Transcribe

Converts speech to text using OpenAI's speech-to-text API.

Use cases:
- Generate accurate transcriptions of audio content
- Create searchable text from audio recordings
- Support multiple languages for transcription
- Enable automated subtitling and captioning

**Tags:** audio, transcription, speech-to-text, stt, whisper

**Fields:**
- **model**: The model to use for transcription. (TranscriptionModel)
- **audio**: The audio file to transcribe (max 25 MB). (AudioRef)
- **language**: The language of the input audio (Language)
- **timestamps**: Whether to return timestamps for the generated text. (bool)
- **prompt**: Optional text to guide the model's style or continue a previous audio segment. (str)
- **temperature**: The sampling temperature between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. (float)


## Translate

Translates speech in audio to English text.

Use cases:
- Translate foreign language audio content to English
- Create English transcripts of multilingual recordings
- Assist non-English speakers in understanding audio content
- Enable cross-language communication in audio formats

**Tags:** audio, translation, speech-to-text, localization

**Fields:**
- **audio**: The audio file to translate. (AudioRef)
- **temperature**: The temperature to use for the translation. (float)


## RealtimeAgent

Stream responses using the official OpenAI Realtime client. Supports optional audio input and streams text chunks.

Uses `AsyncOpenAI().beta.realtime.connect(...)` with the events API:
- Sends session settings via `session.update`
- Adds user input via `conversation.item.create`
- Streams back `response.text.delta` events until `response.done`

**Tags:** realtime, streaming, openai, audio-input, text-output

**Fields:**
- **model** (Model)
- **system**: System instructions for the realtime session (str)
- **chunk**: The audio chunk to use as input. (Chunk)
- **voice**: The voice for the audio output (Voice)
- **speed**: The speed of the model's spoken response (float)
- **temperature**: The temperature for the response (float)

### run

Run the realtime agent with streaming input/output and tools.


**Args:**

- **context (ProcessingContext)**: Workflow execution context.
- **inputs (NodeInputs)**: Streaming inputs (text/audio/chunk).
- **outputs (NodeOutputs)**: Output emitter for streaming chunks and artifacts.
**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None

### should_route_output

Do not route dynamic outputs; they represent tool entry points.
Still route declared outputs like 'text', 'chunk', 'audio'.
**Args:**
- **output_name (str)**

**Returns:** bool


## RealtimeTranscription

Stream microphone or audio input to OpenAI Realtime and emit transcription.
Emits:
- `chunk` Chunk(content=..., done=False) for transcript deltas
- `chunk` Chunk(content="", done=True) to mark segment end
- `text` final aggregated transcript when input ends

**Tags:** 

**Fields:**
- **model**: Model to use (LanguageModel)
- **system**: System instructions (optional) (str)
- **temperature**: Decoding temperature (float)

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## CreateImage

Generates images from textual descriptions.

Use cases:
1. Create custom illustrations for articles or presentations
2. Generate concept art for creative projects
3. Produce visual aids for educational content
4. Design unique marketing visuals or product mockups
5. Explore artistic ideas and styles programmatically

**Tags:** image, t2i, tti, text-to-image, create, generate, picture, photo, art, drawing, illustration

**Fields:**
- **prompt**: The prompt to use. (str)
- **model**: The model to use for image generation. (Model)
- **size**: The size of the image to generate. (Size)
- **background**: The background of the image to generate. (Background)
- **quality**: The quality of the image to generate. (Quality)


## Embedding

Generate vector representations of text for semantic analysis.

Uses OpenAI's embedding models to create dense vector representations of text.
These vectors capture semantic meaning, enabling:
- Semantic search
- Text clustering
- Document classification
- Recommendation systems
- Anomaly detection
- Measuring text similarity and diversity

**Tags:** embeddings, similarity, search, clustering, classification

**Fields:**
- **input** (str)
- **model** (EmbeddingModel)
- **chunk_size** (int)


## WebSearch

🔍 OpenAI Web Search - Searches the web using OpenAI's web search capabilities.
This node uses an OpenAI model equipped with web search functionality
(like gpt-4o with search preview) to answer queries based on current web information.
Requires an OpenAI API key.

**Tags:** 

**Fields:**
- **query**: The search query to execute. (str)


## TelegramBotTrigger

Trigger node that listens for Telegram messages using long polling.
This trigger connects to Telegram using a bot token and emits events
for incoming messages.

**Tags:** 

**Fields:**
- **max_events**: Maximum number of events to process (0 = unlimited) (int)
- **token**: Telegram bot token (str)
- **chat_id**: Optional chat ID to filter messages (int | None)
- **allow_bot_messages**: Include messages authored by bots (bool)
- **include_edited_messages**: Include edited messages (bool)
- **poll_timeout_seconds**: Long polling timeout in seconds (int)
- **poll_interval_seconds**: Delay between polling requests (float)

### cleanup_trigger

Stop polling Telegram for updates.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### setup_trigger

Start polling Telegram for updates.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### wait_for_event

Wait for the next Telegram message event.
**Args:**
- **context (ProcessingContext)**

**Returns:** TelegramBotTriggerOutput | None


## TelegramSendMessage

Node that sends a message to a Telegram chat using a bot token.

**Fields:**
- **token**: Telegram bot token (str)
- **chat_id**: Target chat ID (int)
- **text**: Message text (str)
- **parse_mode**: Optional parse mode (MarkdownV2 or HTML) (str)
- **disable_web_page_preview**: Disable link previews (bool)
- **disable_notification**: Send silently (bool)
- **reply_to_message_id**: Reply to a specific message ID (int | None)


## DiscordBotTrigger

Trigger node that listens for Discord messages from a bot account.
This trigger connects to Discord using a bot token and emits events
for incoming messages.

**Tags:** 

**Fields:**
- **max_events**: Maximum number of events to process (0 = unlimited) (int)
- **token**: Discord bot token (str)
- **channel_id**: Optional channel ID to filter messages (str | None)
- **allow_bot_messages**: Include messages authored by bots (bool)

### cleanup_trigger

Disconnect the Discord bot.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### setup_trigger

Connect the Discord bot and start listening for messages.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### wait_for_event

Wait for the next Discord message event.
**Args:**
- **context (ProcessingContext)**

**Returns:** DiscordBotTriggerOutput | None


## DiscordSendMessage

Node that sends a message to a Discord channel using a bot token.

**Fields:**
- **token**: Discord bot token (str)
- **channel_id**: Target channel ID (str)
- **content**: Message content (str)
- **tts**: Send as text-to-speech (bool)
- **embeds**: Embeds as Discord embed dictionaries (list[dict[str, typing.Any]])


## GoogleFinance

Retrieve financial market data and stock information from Google Finance.

Use cases:
- Track stock prices and market performance
- Monitor financial instruments and indices
- Gather historical market data
- Build financial analysis workflows
- Research investment opportunities

**Tags:** google, finance, stocks, market, serp, trading

**Fields:**
- **query**: Stock symbol or company name to search for (str)
- **window**: Time window for financial data (e.g., '1d', '5d', '1m', '3m', '6m', '1y', '5y') (str)


## GoogleImages

Search Google Images to find visual content or perform reverse image search.

Use cases:
- Find images related to a topic or keyword
- Perform reverse image searches to find sources
- Gather visual assets for projects
- Identify similar or related images
- Research visual trends and styles

**Tags:** google, images, serp, visual, reverse, search

**Fields:**
- **keyword**: Search query or keyword for images (str)
- **image_url**: URL of image for reverse image search (str)
- **num_results**: Maximum number of image results to return (int)


## GoogleJobs

Search Google Jobs for employment opportunities and job listings.

Use cases:
- Find job opportunities matching specific criteria
- Monitor job market trends
- Aggregate job listings for career sites
- Research salary ranges and requirements
- Track hiring patterns in industries

**Tags:** google, jobs, employment, careers, serp, hiring

**Fields:**
- **query**: Job title, skills, or company name to search for (str)
- **location**: Geographic location for job search (str)
- **num_results**: Maximum number of job results to return (int)


## GoogleLens

Analyze images using Google Lens to find visual matches and related content.

Use cases:
- Identify objects, products, or landmarks in images
- Find visually similar images
- Discover where images appear online
- Get information about items shown in photos
- Research products and shopping options

**Tags:** google, lens, visual, image, search, serp, identify

**Fields:**
- **image_url**: URL of the image to analyze with Google Lens (str)
- **num_results**: Maximum number of visual search results to return (int)


## GoogleMaps

Search Google Maps for places, businesses, and get location details.

Use cases:
- Find businesses and points of interest
- Get location information and addresses
- Research geographic areas and neighborhoods
- Build location-based applications
- Gather place reviews and ratings

**Tags:** google, maps, places, locations, serp, geography

**Fields:**
- **query**: Place name, address, or location query (str)
- **num_results**: Maximum number of map results to return (int)


## GoogleNews

Search Google News to retrieve current news articles and headlines.

Use cases:
- Monitor breaking news and current events
- Track news coverage on specific topics
- Gather articles for research and analysis
- Build news aggregation workflows
- Monitor brand mentions in media

**Tags:** google, news, serp, articles, journalism

**Fields:**
- **keyword**: Search query or keyword for news articles (str)
- **num_results**: Maximum number of news results to return (int)


## GoogleSearch

Search Google to retrieve organic search results from the web.

Use cases:
- Find relevant websites and pages on any topic
- Research information and gather sources
- Discover recent content and trending topics
- Gather search engine optimization (SEO) insights
- Automate web research workflows

**Tags:** google, search, serp, web, query

**Fields:**
- **keyword**: Search query or keyword to search for (str)
- **num_results**: Maximum number of results to return (int)


## GoogleShopping

Search Google Shopping for products with filters and pricing information.

Use cases:
- Find products and compare prices
- Research product availability and options
- Monitor pricing trends
- Build price comparison tools
- Gather product information for e-commerce

**Tags:** google, shopping, products, ecommerce, serp, prices

**Fields:**
- **query**: Product name or description to search for (str)
- **country**: Country code for shopping search (e.g., 'us', 'uk', 'ca') (str)
- **min_price**: Minimum price filter for products (int)
- **max_price**: Maximum price filter for products (int)
- **condition**: Product condition filter (e.g., 'new', 'used', 'refurbished') (str)
- **sort_by**: Sort order for results (e.g., 'price_low_to_high', 'price_high_to_low', 'review_score') (str)
- **num_results**: Maximum number of shopping results to return (int)


## ListDocuments

List documents in a directory.

**Tags:** files, list, directory

**Fields:**
- **folder**: Directory to scan (str)
- **pattern**: File pattern to match (e.g. *.txt) (str)
- **recursive**: Search subdirectories (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.document.ListDocuments.OutputType, NoneType]


## LoadDocumentFile

Read a document from disk.

**Tags:** files, document, read, input, load, file

**Fields:**
- **path**: Path to the document to read (str)


## SaveDocumentFile

Write a document to disk.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** files, document, write, output, save, file

**Fields:**
- **document**: The document to save (DocumentRef)
- **folder**: Folder where the file will be saved (str)
- **filename**: Name of the file to save. Supports strftime format codes. (str)


## SplitDocument

Split text semantically.

**Tags:** chroma, embedding, collection, RAG, index, text, markdown, semantic

**Fields:**
- **embed_model**: Embedding model to use (LanguageModel)
- **document**: Document ID to associate with the text content (DocumentRef)
- **buffer_size**: Buffer size for semantic splitting (int)
- **threshold**: Breakpoint percentile threshold for semantic splitting (int)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.document.SplitDocument.OutputType, NoneType]


## SplitHTML

Split HTML content into semantic chunks based on HTML tags.

**Tags:** html, text, semantic, tags, parsing

**Fields:**
- **document**: Document ID to associate with the HTML content (DocumentRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.document.SplitHTML.OutputType, NoneType]


## SplitJSON

Split JSON content into semantic chunks.

**Tags:** json, parsing, semantic, structured

**Fields:**
- **document**: Document ID to associate with the JSON content (DocumentRef)
- **include_metadata**: Whether to include metadata in nodes (bool)
- **include_prev_next_rel**: Whether to include prev/next relationships (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.document.SplitJSON.OutputType, NoneType]


## SplitMarkdown

Splits markdown text by headers while preserving header hierarchy in metadata.

Use cases:
- Splitting markdown documentation while preserving structure
- Processing markdown files for semantic search
- Creating context-aware chunks from markdown content

**Tags:** markdown, split, headers

**Fields:**
- **document** (DocumentRef)
- **headers_to_split_on**: List of tuples containing (header_symbol, header_name) (list[tuple[str, str]])
- **strip_headers**: Whether to remove headers from the output content (bool)
- **return_each_line**: Whether to split into individual lines instead of header sections (bool)
- **chunk_size**: Optional maximum chunk size for further splitting (int)
- **chunk_overlap**: Overlap size when using chunk_size (int)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.document.SplitMarkdown.OutputType, NoneType]


## SplitRecursively

Splits text recursively using LangChain's RecursiveCharacterTextSplitter.

Use cases:
- Splitting documents while preserving semantic relationships
- Creating chunks for language model processing
- Handling text in languages with/without word boundaries

**Tags:** text, split, chunks

**Fields:**
- **document** (DocumentRef)
- **chunk_size**: Maximum size of each chunk in characters (int)
- **chunk_overlap**: Number of characters to overlap between chunks (int)
- **separators**: List of separators to use for splitting, in order of preference (list[str])

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.document.SplitRecursively.OutputType, NoneType]


## Append

Adds a value to the end of a list.

Use cases:
- Grow a list dynamically
- Add new elements to an existing list
- Implement a stack-like structure

**Tags:** list, add, insert, extend

**Fields:**
- **values** (list[typing.Any])
- **value**: The value to append to the list. (Any)


## Average

Calculates the arithmetic mean of a list of numbers.

Use cases:
- Find average value
- Calculate mean of numeric data

**Tags:** list, average, mean, aggregate, math

**Fields:**
- **values** (list[float])


## Chunk

Splits a list into smaller chunks of specified size.

Use cases:
- Batch processing
- Pagination
- Creating sublists of fixed size

**Tags:** list, chunk, split, group

**Fields:**
- **values** (list[typing.Any])
- **chunk_size** (int)


## Dedupe

Removes duplicate elements from a list, ensuring uniqueness.

Use cases:
- Remove redundant entries
- Create a set-like structure
- Ensure list elements are unique

**Tags:** list, unique, distinct, deduplicate

**Fields:**
- **values** (list[typing.Any])


## Difference

Finds elements that exist in first list but not in second list.

Use cases:
- Find unique elements in one list
- Remove items present in another list
- Identify distinct elements

**Tags:** list, set, difference, subtract

**Fields:**
- **list1** (list[typing.Any])
- **list2** (list[typing.Any])


## Extend

Merges one list into another, extending the original list.

Use cases:
- Combine multiple lists
- Add all elements from one list to another

**Tags:** list, merge, concatenate, combine

**Fields:**
- **values** (list[typing.Any])
- **other_values** (list[typing.Any])


## Flatten

Flattens a nested list structure into a single flat list.

Use cases:
- Convert nested lists into a single flat list
- Simplify complex list structures
- Process hierarchical data as a sequence

Examples:
[[1, 2], [3, 4]] -> [1, 2, 3, 4]
[[1, [2, 3]], [4, [5, 6]]] -> [1, 2, 3, 4, 5, 6]

**Tags:** list, flatten, nested, structure

**Fields:**
- **values** (list[typing.Any])
- **max_depth** (int)


## GenerateSequence

Iterates over a sequence of numbers.

**Tags:** list, range, sequence, numbers

**Fields:**
- **start** (int)
- **stop** (int)
- **step** (int)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.list.GenerateSequence.OutputType, NoneType]


## GetElement

Retrieves a single value from a list at a specific index.

Use cases:
- Access a specific element by position
- Implement array-like indexing
- Extract the first or last element

**Tags:** list, get, extract, value

**Fields:**
- **values** (list[typing.Any])
- **index** (int)


## Intersection

Finds common elements between two lists.

Use cases:
- Find elements present in both lists
- Identify shared items between collections
- Filter for matching elements

**Tags:** list, set, intersection, common

**Fields:**
- **list1** (list[typing.Any])
- **list2** (list[typing.Any])


## Length

Calculates the length of a list.

Use cases:
- Determine the number of elements in a list
- Check if a list is empty
- Validate list size constraints

**Tags:** list, count, size

**Fields:**
- **values** (list[typing.Any])


## ListRange

Generates a list of integers within a specified range.

Use cases:
- Create numbered lists
- Generate index sequences
- Produce arithmetic progressions

**Tags:** list, range, sequence, numbers

**Fields:**
- **start** (int)
- **stop** (int)
- **step** (int)


## Maximum

Finds the largest value in a list of numbers.

Use cases:
- Find highest value
- Get largest number in dataset

**Tags:** list, max, maximum, aggregate, math

**Fields:**
- **values** (list[float])


## Minimum

Finds the smallest value in a list of numbers.

Use cases:
- Find lowest value
- Get smallest number in dataset

**Tags:** list, min, minimum, aggregate, math

**Fields:**
- **values** (list[float])


## Product

Calculates the product of all numbers in a list.

Use cases:
- Multiply all numbers together
- Calculate compound values

**Tags:** list, product, multiply, aggregate, math

**Fields:**
- **values** (list[float])


## Randomize

Randomly shuffles the elements of a list.

Use cases:
- Randomize the order of items in a playlist
- Implement random sampling without replacement
- Create randomized data sets for testing

**Tags:** list, shuffle, random, order

**Fields:**
- **values** (list[typing.Any])


## Reverse

Inverts the order of elements in a list.

Use cases:
- Reverse the order of a sequence

**Tags:** list, reverse, invert, flip

**Fields:**
- **values** (list[typing.Any])


## SaveList

Saves a list to a text file, placing each element on a new line.

Use cases:
- Export list data to a file
- Create a simple text-based database
- Generate line-separated output

**Tags:** list, save, file, serialize

**Fields:**
- **values** (list[typing.Any])
- **name**: 
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**


## SelectElements

Selects specific values from a list using index positions.

Use cases:
- Pick specific elements by their positions
- Rearrange list elements
- Create a new list from selected indices

**Tags:** list, select, index, extract

**Fields:**
- **values** (list[typing.Any])
- **indices** (list[int])


## Slice

Extracts a subset from a list using start, stop, and step indices.

Use cases:
- Get a portion of a list
- Implement pagination
- Extract every nth element

**Tags:** list, slice, subset, extract

**Fields:**
- **values** (list[typing.Any])
- **start** (int)
- **stop** (int)
- **step** (int)


## Sort

Sorts the elements of a list in ascending or descending order.

Use cases:
- Organize data in a specific order
- Prepare data for binary search or other algorithms
- Rank items based on their values

**Tags:** list, sort, order, arrange

**Fields:**
- **values** (list[typing.Any])
- **order** (SortOrder)


## Sum

Calculates the sum of a list of numbers.

Use cases:
- Calculate total of numeric values
- Add up all elements in a list

**Tags:** list, sum, aggregate, math

**Fields:**
- **values** (list[float])


## Union

Combines unique elements from two lists.

Use cases:
- Merge lists while removing duplicates
- Combine collections uniquely
- Create comprehensive set of items

**Tags:** list, set, union, combine

**Fields:**
- **list1** (list[typing.Any])
- **list2** (list[typing.Any])


## ChartGenerator

LLM Agent to create Plotly Express charts based on natural language descriptions.

Use cases:
- Generating interactive charts from natural language descriptions
- Creating data visualizations with minimal configuration
- Converting data analysis requirements into visual representations

**Tags:** llm, data visualization, charts

**Fields:**
- **model**: The model to use for chart generation. (LanguageModel)
- **prompt**: Natural language description of the desired chart (str)
- **data**: The data to visualize (DataframeRef)
- **max_tokens**: The maximum number of tokens to generate. (int)


## DataGenerator

LLM Agent to create a dataframe based on a user prompt.

Use cases:
- Generating structured data from natural language descriptions
- Creating sample datasets for testing or demonstration
- Converting unstructured text into tabular format

**Tags:** llm, dataframe creation, data structuring

**Fields:**
- **model**: The model to use for data generation. (LanguageModel)
- **prompt**: The user prompt (str)
- **input_text**: The input text to be analyzed by the agent. (str)
- **max_tokens**: The maximum number of tokens to generate. (int)
- **columns**: The columns to use in the dataframe. (RecordType)

### gen_process

Streaming generation that yields individual records as they are generated
and a final dataframe once all records are ready.
**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.generators.DataGenerator.OutputType, NoneType]


## ListGenerator

LLM Agent to create a stream of strings based on a user prompt.

Use cases:
- Generating text from natural language descriptions
- Streaming responses from an LLM

**Tags:** llm, text streaming

**Fields:**
- **model**: The model to use for string generation. (LanguageModel)
- **prompt**: The user prompt (str)
- **input_text**: The input text to be analyzed by the agent. (str)
- **max_tokens**: The maximum number of tokens to generate. (int)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.generators.ListGenerator.OutputType, NoneType]


## SVGGenerator

LLM Agent to create SVG elements based on user prompts.

Use cases:
- Creating vector graphics from text descriptions
- Generating scalable illustrations
- Creating custom icons and diagrams

**Tags:** svg, generator, vector, graphics

**Fields:**
- **model**: The language model to use for SVG generation. (LanguageModel)
- **prompt**: The user prompt for SVG generation (str)
- **image**: Image to use for generation (ImageRef)
- **audio**: Audio to use for generation (AudioRef)
- **max_tokens**: The maximum number of tokens to generate. (int)


## StructuredOutputGenerator

Generate structured JSON objects from instructions using LLM providers.

Specialized for creating structured information:
- Generating JSON that follows dynamic schemas
- Fabricating records from requirements and guidance
- Simulating sample data for downstream workflows
- Producing consistent structured outputs for testing

**Tags:** data-generation, structured-data, json, synthesis

**Fields:**
- **system_prompt**: The system prompt guiding JSON generation. (str)
- **model**: Model to use for structured generation. (LanguageModel)
- **instructions**: Detailed instructions for the structured output. (str)
- **context**: Optional context to ground the generation. (str)
- **max_tokens**: The maximum number of tokens to generate. (int)
- **context_window** (int)


## AudioMixer

Mix up to 5 audio tracks together with individual volume controls.

Use cases:
- Mix multiple audio tracks into a single output
- Create layered soundscapes
- Combine music, voice, and sound effects
- Adjust individual track volumes

**Tags:** audio, mix, volume, combine, blend, layer, add, overlay

**Fields:**
- **track1**: First audio track to mix. (AudioRef)
- **track2**: Second audio track to mix. (AudioRef)
- **track3**: Third audio track to mix. (AudioRef)
- **track4**: Fourth audio track to mix. (AudioRef)
- **track5**: Fifth audio track to mix. (AudioRef)
- **volume1**: Volume for track 1. 1.0 is original volume. (float)
- **volume2**: Volume for track 2. 1.0 is original volume. (float)
- **volume3**: Volume for track 3. 1.0 is original volume. (float)
- **volume4**: Volume for track 4. 1.0 is original volume. (float)
- **volume5**: Volume for track 5. 1.0 is original volume. (float)


## AudioToNumpy

Convert audio to numpy array for processing.

Use cases:
- Prepare audio for custom processing
- Convert audio for machine learning models
- Extract raw audio data for analysis

**Tags:** audio, numpy, convert, array

**Fields:**
- **audio**: The audio to convert to numpy. (AudioRef)


## Concat

Concatenates two audio files together.

Use cases:
- Combine multiple audio clips into a single file
- Create longer audio tracks from shorter segments

**Tags:** audio, edit, join, +

**Fields:**
- **a**: The first audio file. (AudioRef)
- **b**: The second audio file. (AudioRef)


## ConcatList

Concatenates multiple audio files together in sequence.

Use cases:
- Combine multiple audio clips into a single file
- Create longer audio tracks from multiple segments
- Chain multiple audio files in order

**Tags:** audio, edit, join, multiple, +

**Fields:**
- **audio_files**: List of audio files to concatenate in sequence. (list[nodetool.metadata.types.AudioRef])


## ConvertToArray

Converts an audio file to a Array for further processing.

Use cases:
- Prepare audio data for machine learning models
- Enable signal processing operations on audio
- Convert audio to a format suitable for spectral analysisr

**Tags:** audio, conversion, tensor

**Fields:**
- **audio**: The audio file to convert to a tensor. (AudioRef)


## CreateSilence

Creates a silent audio file with a specified duration.

Use cases:
- Generate placeholder audio files
- Create audio segments for padding or spacing
- Add silence to the beginning or end of audio files

**Tags:** audio, silence, empty

**Fields:**
- **duration**: The duration of the silence in seconds. (float)


## FadeIn

Applies a fade-in effect to the beginning of an audio file.

Use cases:
- Create smooth introductions to audio tracks
- Gradually increase volume at the start of a clip

**Tags:** audio, edit, transition

**Fields:**
- **audio**: The audio file to apply fade-in to. (AudioRef)
- **duration**: Duration of the fade-in effect in seconds. (float)


## FadeOut

Applies a fade-out effect to the end of an audio file.

Use cases:
- Create smooth endings to audio tracks
- Gradually decrease volume at the end of a clip

**Tags:** audio, edit, transition

**Fields:**
- **audio**: The audio file to apply fade-out to. (AudioRef)
- **duration**: Duration of the fade-out effect in seconds. (float)


## LoadAudioAssets

Load audio files from an asset folder.

**Tags:** load, audio, file, import

**Fields:**
- **folder**: The asset folder to load the audio files from. (FolderRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.audio.LoadAudioAssets.OutputType, NoneType]


## LoadAudioFile

Read an audio file from disk.

Use cases:
- Load audio for processing
- Import sound files for editing
- Read audio assets for a workflow

**Tags:** audio, input, load, file

**Fields:**
- **path**: Path to the audio file to read (str)


## LoadAudioFolder

Load all audio files from a folder, optionally including subfolders.

Use cases:
- Batch import audio for processing
- Build datasets from a directory tree
- Iterate over audio collections

**Tags:** audio, load, folder, files

**Fields:**
- **folder**: Folder to scan for audio files (str)
- **include_subdirectories**: Include audio in subfolders (bool)
- **extensions**: Audio file extensions to include (list[str])

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.audio.LoadAudioFolder.OutputType, NoneType]


## MonoToStereo

Converts a mono audio signal to stereo.

Use cases:
- Expand mono recordings for stereo playback systems
- Prepare audio for further stereo processing

**Tags:** audio, convert, channels

**Fields:**
- **audio**: The mono audio file to convert. (AudioRef)


## Normalize

Normalizes the volume of an audio file.

Use cases:
- Ensure consistent volume across multiple audio files
- Adjust overall volume level before further processing

**Tags:** audio, fix, dynamics, volume

**Fields:**
- **audio**: The audio file to normalize. (AudioRef)


## NumpyToAudio

Convert numpy array to audio.

Use cases:
- Convert processed audio data back to audio format
- Create audio from machine learning model outputs
- Generate audio from synthesized waveforms

**Tags:** audio, numpy, convert

**Fields:**
- **array**: The numpy array to convert to audio. (NPArray)
- **sample_rate**: Sample rate in Hz. (int)
- **channels**: Number of audio channels (1 or 2). (int)


## OverlayAudio

Overlays two audio files together.

Use cases:
- Mix background music with voice recording
- Layer sound effects over an existing audio track

**Tags:** audio, edit, transform

**Fields:**
- **a**: The first audio file. (AudioRef)
- **b**: The second audio file. (AudioRef)


## RemoveSilence

Removes or shortens silence in an audio file with smooth transitions.

Use cases:
- Trim silent parts from beginning/end of recordings
- Remove or shorten long pauses between speech segments
- Apply crossfade for smooth transitions

**Tags:** audio, edit, clean

**Fields:**
- **audio**: The audio file to process. (AudioRef)
- **min_length**: Minimum length of silence to be processed (in milliseconds). (int)
- **threshold**: Silence threshold in dB (relative to full scale). Higher values detect more silence. (int)
- **reduction_factor**: Factor to reduce silent parts (0.0 to 1.0). 0.0 keeps silence as is, 1.0 removes it completely. (float)
- **crossfade**: Duration of crossfade in milliseconds to apply between segments for smooth transitions. (int)
- **min_silence_between_parts**: Minimum silence duration in milliseconds to maintain between non-silent segments (int)


## Repeat

Loops an audio file a specified number of times.

Use cases:
- Create repeating background sounds or music
- Extend short audio clips to fill longer durations
- Generate rhythmic patterns from short samples

**Tags:** audio, edit, repeat

**Fields:**
- **audio**: The audio file to loop. (AudioRef)
- **loops**: Number of times to loop the audio. Minimum 1 (plays once), maximum 100. (int)


## Reverse

Reverses an audio file.

Use cases:
- Create reverse audio effects
- Generate backwards speech or music

**Tags:** audio, edit, transform

**Fields:**
- **audio**: The audio file to reverse. (AudioRef)


## SaveAudio

Save an audio file to a specified asset folder.

Use cases:
- Save generated audio files with timestamps
- Organize outputs into specific folders
- Create backups of generated audio

**Tags:** audio, folder, name

**Fields:**
- **audio** (AudioRef)
- **folder**: The asset folder to save the audio file to.  (FolderRef)
- **name**: 
        The name of the audio file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**


## SaveAudioFile

Write an audio file to disk.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** audio, output, save, file

**Fields:**
- **audio**: The audio to save (AudioRef)
- **folder**: Folder where the file will be saved (str)
- **filename**: 
        Name of the file to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## SliceAudio

Extracts a section of an audio file.

Use cases:
- Cut out a specific clip from a longer audio file
- Remove unwanted portions from beginning or end

**Tags:** audio, edit, trim

**Fields:**
- **audio**: The audio file. (AudioRef)
- **start**: The start time in seconds. (float)
- **end**: The end time in seconds. (float)


## StereoToMono

Converts a stereo audio signal to mono.

Use cases:
- Reduce file size for mono-only applications
- Simplify audio for certain processing tasks

**Tags:** audio, convert, channels

**Fields:**
- **audio**: The stereo audio file to convert. (AudioRef)
- **method**: Method to use for conversion: 'average', 'left', or 'right'. (str)


## TextToSpeech

Generate speech audio from text using any supported TTS provider.
audio, generation, AI, text-to-speech, tts, voice

Use cases:
- Create voiceovers for videos and presentations
- Generate natural-sounding narration for content
- Build voice assistants and chatbots
- Convert written content to audio format
- Create accessible audio versions of text

**Tags:** Automatically routes to the appropriate backend (OpenAI, HuggingFace, MLX).

**Fields:**
- **model**: The text-to-speech model to use (TTSModel)
- **text**: Text to convert to speech (str)
- **speed**: Speech speed multiplier (0.25 to 4.0) (float)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.audio.TextToSpeech.OutputType, NoneType]


## Trim

Trim an audio file to a specified duration.

Use cases:
- Remove silence from the beginning or end of audio files
- Extract specific segments from audio files
- Prepare audio data for machine learning models

**Tags:** audio, trim, cut

**Fields:**
- **audio**: The audio file to trim. (AudioRef)
- **start**: The start time of the trimmed audio in seconds. (float)
- **end**: The end time of the trimmed audio in seconds. (float)


## ArrayOutput

Output node for generic array data, typically numerical ('NPArray').

Use cases:
- Outputting results from machine learning models (e.g., embeddings, predictions).
- Representing complex numerical data structures.
- Passing arrays of numbers between processing steps.

**Tags:** array, numerical, list, tensor, vector, matrix

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (NPArray)
- **description**: The description of the output for the workflow. (str)


## AudioOutput

Output node for audio content references ('AudioRef').

Use cases:
- Displaying or returning processed or generated audio.
- Passing audio data (as an 'AudioRef') between workflow nodes.
- Returning results of audio analysis (e.g., transcription reference, audio features).

**Tags:** audio, sound, media, voice, speech, asset, reference

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (AudioRef)
- **description**: The description of the output for the workflow. (str)


## BooleanOutput

Output node for a single boolean value.

Use cases:
- Returning binary results (yes/no, true/false)
- Controlling conditional logic in workflows
- Indicating success/failure of operations

**Tags:** boolean, true, false, flag, condition, flow-control, branch, else, switch, toggle

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (bool)
- **description**: The description of the output for the workflow. (str)


## DataframeOutput

Output node for structured data references, typically tabular ('DataframeRef').

Use cases:
- Outputting tabular data results from analysis or queries.
- Passing structured datasets between processing or analysis steps.
- Displaying data in a table format or making it available for download.

**Tags:** dataframe, table, structured, csv, tabular_data, rows, columns

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (DataframeRef)
- **description**: The description of the output for the workflow. (str)


## DictionaryOutput

Output node for key-value pair data (dictionary).

Use cases:
- Returning multiple named values as a single structured output.
- Passing complex data structures or configurations between nodes.
- Organizing heterogeneous output data into a named map.

**Tags:** dictionary, key-value, mapping, object, json_object, struct

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (dict[str, typing.Any])
- **description**: The description of the output for the workflow. (str)


## DocumentOutput

Output node for document content references ('DocumentRef').

Use cases:
- Displaying or returning processed or generated documents.
- Passing document data (as a 'DocumentRef') between workflow nodes.
- Returning results of document analysis or manipulation.

**Tags:** document, file, pdf, text_file, asset, reference

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (DocumentRef)
- **description**: The description of the output for the workflow. (str)


## FilePathOutput

Output node for a file path.

**Tags:** file, path, file_path

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (FilePath)
- **description**: The description of the output for the workflow. (str)


## FloatOutput

Output node for a single float value.

Use cases:
- Returning decimal results (e.g. percentages, ratios)
- Passing floating-point parameters between nodes
- Displaying numeric metrics with decimal precision

**Tags:** float, decimal, number

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (float)
- **description**: The description of the output for the workflow. (str)


## FolderPathOutput

Output node for a folder path.

**Tags:** folder, path, folder_path

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (FolderPath)
- **description**: The description of the output for the workflow. (str)


## ImageOutput

Output node for a single image reference ('ImageRef').

Use cases:
- Displaying a single processed or generated image.
- Passing image data (as an 'ImageRef') between workflow nodes.
- Returning image analysis results encapsulated in an 'ImageRef'.

**Tags:** image, picture, visual, asset, reference

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (ImageRef)
- **description**: The description of the output for the workflow. (str)


## IntegerOutput

Output node for a single integer value.

Use cases:
- Returning numeric results (e.g. counts, indices)
- Passing integer parameters between nodes
- Displaying numeric metrics

**Tags:** integer, number, count

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (int)
- **description**: The description of the output for the workflow. (str)


## ListOutput

Output node for a list of arbitrary values.

Use cases:
- Returning multiple results from a workflow
- Aggregating outputs from multiple nodes

**Tags:** list, output, any

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (list[typing.Any])
- **description**: The description of the output for the workflow. (str)


## StringOutput

Output node for a string value.

Use cases:
- Returning short text results or messages.
- Passing concise string parameters or identifiers between nodes.
- Displaying brief textual outputs.
- For multi-line text or structured document content, use appropriate output nodes if available or consider how data is structured.

**Tags:** string, text, output, label, name

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (str)
- **description**: The description of the output for the workflow. (str)


## VideoOutput

Output node for video content references ('VideoRef').

Use cases:
- Displaying processed or generated video content.
- Passing video data (as a 'VideoRef') between workflow steps.
- Returning results of video analysis encapsulated in a 'VideoRef'.

**Tags:** video, media, clip, asset, reference

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (VideoRef)
- **description**: The description of the output for the workflow. (str)


## ArgMax

Returns the label associated with the highest value in a dictionary.

Use cases:
- Get the most likely class from classification probabilities
- Find the category with highest score
- Identify the winner in a voting/ranking system

**Tags:** dictionary, maximum, label, argmax

**Fields:**
- **scores**: Dictionary mapping labels to their corresponding scores/values (dict[str, float])


## Combine

Merges two dictionaries, with second dictionary values taking precedence.

Use cases:
- Combine default and custom configurations
- Merge partial updates with existing data
- Create aggregate data structures

**Tags:** dictionary, merge, update, +, add, concatenate

**Fields:**
- **dict_a** (dict[str, typing.Any])
- **dict_b** (dict[str, typing.Any])


## Filter

Creates a new dictionary with only specified keys from the input.

Use cases:
- Extract relevant fields from a larger data structure
- Implement data access controls
- Prepare specific data subsets for processing

**Tags:** dictionary, filter, select

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **keys** (list[str])


## FilterDictByNumber

Filters a stream of dictionaries based on numeric values for a specified key.

Use cases:
- Filter dictionaries by numeric comparisons (greater than, less than, equal to)
- Filter records with even/odd numeric values

**Tags:** filter, dictionary, numbers, numeric, stream

**Fields:**
- **value**: Input dictionary stream (dict)
- **key**: The dictionary key to check (str)
- **filter_type** (FilterDictNumberType)
- **compare_value**: Comparison value (float)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.dictionary.FilterDictByNumber.OutputType, NoneType]


## FilterDictByQuery

Filter a stream of dictionary objects based on a pandas query condition.

Basic Operators:
- Comparison: >, <, >=, <=, ==, !=
- Logical: and, or, not
- Membership: in, not in

Use cases:
- Filter dictionary objects based on complex criteria
- Extract subset of data meeting specific conditions

**Tags:** filter, query, condition, dictionary, stream

**Fields:**
- **value**: Input dictionary stream (dict)
- **condition**: The filtering condition using pandas query syntax. (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.dictionary.FilterDictByQuery.OutputType, NoneType]


## FilterDictByRange

Filters a stream of dictionaries based on a numeric range for a specified key.

Use cases:
- Filter records based on numeric ranges (e.g., price range, age range)
- Find entries with values within specified bounds

**Tags:** filter, dictionary, range, between, stream

**Fields:**
- **value**: Input dictionary stream (dict)
- **key**: The dictionary key to check for the range (str)
- **min_value**: The minimum value (inclusive) of the range (float)
- **max_value**: The maximum value (inclusive) of the range (float)
- **inclusive**: If True, includes the min and max values in the results (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.dictionary.FilterDictByRange.OutputType, NoneType]


## FilterDictByValue

Filters a stream of dictionaries based on their values using various criteria.

Use cases:
- Filter dictionaries by value content
- Filter dictionaries by value type
- Filter dictionaries by value patterns

**Tags:** filter, dictionary, values, stream

**Fields:**
- **value**: Input dictionary stream (dict)
- **key**: The dictionary key to check (str)
- **filter_type**: The type of filter to apply (FilterType)
- **criteria**: The filtering criteria (text to match, type name, or length as string) (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.dictionary.FilterDictByValue.OutputType, NoneType]


## FilterDictRegex

Filters a stream of dictionaries using regular expressions on specified keys.

Use cases:
- Filter dictionaries with values matching complex patterns
- Search for dictionaries containing emails, dates, or specific formats

**Tags:** filter, regex, dictionary, pattern, stream

**Fields:**
- **value**: Input dictionary stream (dict)
- **key**: The dictionary key to check (str)
- **pattern**: The regex pattern (str)
- **full_match**: Full match or partial (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.dictionary.FilterDictRegex.OutputType, NoneType]


## GetValue

Retrieves a value from a dictionary using a specified key.

Use cases:
- Access a specific item in a configuration dictionary
- Retrieve a value from a parsed JSON object
- Extract a particular field from a data structure

**Tags:** dictionary, get, value, key

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **key** (str)
- **default** (Any)


## LoadCSVFile

Read a CSV file from disk.

**Tags:** files, csv, read, input, load, file

**Fields:**
- **path**: Path to the CSV file to read (str)


## MakeDictionary

Creates a simple dictionary with up to three key-value pairs.

Use cases:
- Create configuration entries
- Initialize simple data structures
- Build basic key-value mappings

**Tags:** dictionary, create, simple

**Fields:**


## ParseJSON

Parses a JSON string into a Python dictionary.

Use cases:
- Process API responses
- Load configuration files
- Deserialize stored data

**Tags:** json, parse, dictionary

**Fields:**
- **json_string** (str)


## ReduceDictionaries

Reduces a list of dictionaries into one dictionary based on a specified key field.

Use cases:
- Aggregate data by a specific field
- Create summary dictionaries from list of records
- Combine multiple data points into a single structure

**Tags:** dictionary, reduce, aggregate

**Fields:**
- **dictionaries**: List of dictionaries to be reduced (list[dict[str, typing.Any]])
- **key_field**: The field to use as the key in the resulting dictionary (str)
- **value_field**: Optional field to use as the value. If not specified, the entire dictionary (minus the key field) will be used as the value. (str)
- **conflict_resolution**: How to handle conflicts when the same key appears multiple times (ConflictResolution)


## Remove

Removes a key-value pair from a dictionary.

Use cases:
- Delete a specific configuration option
- Remove sensitive information before processing
- Clean up temporary entries in a data structure

**Tags:** dictionary, remove, delete

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **key** (str)


## SaveCSVFile

Write a list of dictionaries to a CSV file.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** files, csv, write, output, save, file

**Fields:**
- **data**: list of dictionaries to write to CSV (list[dict])
- **folder**: Folder where the file will be saved (str)
- **filename**: Name of the CSV file to save. Supports strftime format codes. (str)


## Update

Updates a dictionary with new key-value pairs.

Use cases:
- Extend a configuration with additional settings
- Add new entries to a cache or lookup table
- Merge user input with existing data

**Tags:** dictionary, add, update

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **new_pairs** (dict[str, typing.Any])


## Zip

Creates a dictionary from parallel lists of keys and values.

Use cases:
- Convert separate data columns into key-value pairs
- Create lookups from parallel data structures
- Transform list data into associative arrays

**Tags:** dictionary, create, zip

**Fields:**
- **keys** (list[typing.Any])
- **values** (list[typing.Any])


## Agent

Generate natural language responses using LLM providers and streams output.

**Tags:** llm, text-generation, chatbot, question-answering, streaming

**Fields:**
- **model**: Model to use for execution (LanguageModel)
- **system**: The system prompt for the LLM (str)
- **prompt**: The prompt for the LLM (str)
- **image**: The image to analyze (ImageRef)
- **audio**: The audio to analyze (AudioRef)
- **history**: The messages for the LLM (list[nodetool.metadata.types.Message])
- **thread_id**: Optional thread ID for persistent conversation history. If provided, messages will be loaded from and saved to this thread. (str | None)
- **max_tokens** (int)
- **context_window** (int)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.agents.Agent.OutputType, NoneType]

### should_route_output

Do not route dynamic outputs; they represent tool entry points.
Still route declared outputs like 'text', 'chunk', 'audio'.
**Args:**
- **output_name (str)**

**Returns:** bool


## Classifier

Classify text into predefined or dynamic categories using LLM.

Use cases:
- Sentiment analysis
- Topic classification
- Intent detection
- Content categorization

**Tags:** classification, nlp, categorization

**Fields:**
- **system_prompt**: The system prompt for the classifier (str)
- **model**: Model to use for classification (LanguageModel)
- **text**: Text to classify (str)
- **image**: Optional image to classify in context (ImageRef)
- **audio**: Optional audio to classify in context (AudioRef)
- **categories**: List of possible categories. If empty, LLM will determine categories. (list[str])
- **context_window** (int)


## CreateThread

Create a new conversation thread and return its ID.

Use this to seed a thread_id that downstream Agent nodes can reuse for
persistent history across the graph or multiple runs.

**Tags:** threads, chat, conversation, context

**Fields:**
- **title**: Optional title for the new thread (str | None)
- **thread_id**: Optional custom thread ID. If provided and owned by the user, it will be reused; otherwise a new thread is created. (str)


## Extractor

Extract structured data from text content using LLM providers.

Specialized for extracting structured information:
- Converting unstructured text into structured data
- Identifying and extracting specific fields from documents
- Parsing text according to predefined schemas
- Creating structured records from natural language content

**Tags:** data-extraction, structured-data, nlp, parsing

**Fields:**
- **system_prompt**: The system prompt for the data extractor (str)
- **model**: Model to use for data extraction (LanguageModel)
- **text**: The text to extract data from (str)
- **image**: Optional image to assist extraction (ImageRef)
- **audio**: Optional audio to assist extraction (AudioRef)
- **context_window** (int)


## ResearchAgent

Autonomous research agent that gathers information from the web and synthesizes findings.

Uses dynamic outputs to define the structure of research results.
The agent will:
- Search the web for relevant information
- Browse and extract content from web pages
- Organize findings in the workspace
- Return structured results matching your output schema

Perfect for:
- Market research and competitive analysis
- Literature reviews and fact-finding
- Data collection from multiple sources
- Automated research workflows

**Tags:** research, web-search, data-gathering, agent, automation

**Fields:**
- **objective**: The research objective or question to investigate (str)
- **model**: Model to use for research and synthesis (LanguageModel)
- **system_prompt**: System prompt guiding the agent's research behavior (str)
- **tools**: Additional research tools to enable (workspace tools are always included) (list[nodetool.metadata.types.ToolName])
- **max_tokens**: Maximum tokens for agent responses (int)
- **context_window**: Context window size (int)


## Summarizer

Generate concise summaries of text content using LLM providers with streaming output.

Specialized for creating high-quality summaries with real-time streaming:
- Condensing long documents into key points
- Creating executive summaries with live output
- Extracting main ideas from text as they're generated
- Maintaining factual accuracy while reducing length

**Tags:** text, summarization, nlp, content, streaming

**Fields:**
- **system_prompt**: The system prompt for the summarizer (str)
- **model**: Model to use for summarization (LanguageModel)
- **text**: The text to summarize (str)
- **image**: Optional image to condition the summary (ImageRef)
- **audio**: Optional audio to condition the summary (AudioRef)
- **context_window** (int)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.agents.Summarizer.OutputType, NoneType]


## FilterNumber

Filters a stream of numbers based on various numerical conditions.

Use cases:
- Filter numbers by comparison (greater than, less than, equal to)
- Filter even/odd numbers
- Filter positive/negative numbers

**Tags:** filter, numbers, numeric, stream

**Fields:**
- **value**: Input number stream (float)
- **filter_type**: The type of filter to apply (FilterNumberType)
- **compare_value**: The comparison value (for greater_than, less_than, equal_to) (float)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.numbers.FilterNumber.OutputType, NoneType]


## FilterNumberRange

Filters a stream of numbers to find values within a specified range.

Use cases:
- Find numbers within a specific range
- Filter data points within bounds
- Implement range-based filtering

**Tags:** filter, numbers, range, between, stream

**Fields:**
- **value**: Input number stream (float)
- **min_value**: Minimum value (float)
- **max_value**: Maximum value (float)
- **inclusive**: Inclusive bounds (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.numbers.FilterNumberRange.OutputType, NoneType]


## AddColumn

Add list of values as new column to dataframe.

Use cases:
- Incorporate external data into existing dataframe
- Add calculated results as new column
- Augment dataframe with additional features

**Tags:** dataframe, column, list

**Fields:**
- **dataframe**: Dataframe object to add a new column to. (DataframeRef)
- **column_name**: The name of the new column to be added to the dataframe. (str)
- **values**: A list of any type of elements which will be the new column's values. (list[typing.Any])


## Aggregate

Aggregate dataframe by one or more columns.

Use cases:
- Prepare data for aggregation operations
- Analyze data by categories
- Create summary statistics by groups

**Tags:** aggregate, groupby, group, sum, mean, count, min, max, std, var, median, first, last

**Fields:**
- **dataframe**: The DataFrame to group. (DataframeRef)
- **columns**: Comma-separated column names to group by. (str)
- **aggregation**: Aggregation function: sum, mean, count, min, max, std, var, median, first, last (str)


## Append

Append two dataframes along rows.

Use cases:
- Combine data from multiple time periods
- Merge datasets with same structure
- Aggregate data from different sources

**Tags:** append, concat, rows

**Fields:**
- **dataframe_a**: First DataFrame to be appended. (DataframeRef)
- **dataframe_b**: Second DataFrame to be appended. (DataframeRef)


## DropDuplicates

Remove duplicate rows from dataframe.

Use cases:
- Clean dataset by removing redundant entries
- Ensure data integrity in analysis
- Prepare data for unique value operations

**Tags:** duplicates, unique, clean

**Fields:**
- **df**: The input DataFrame. (DataframeRef)


## DropNA

Remove rows with NA values from dataframe.

Use cases:
- Clean dataset by removing incomplete entries
- Prepare data for analysis requiring complete cases
- Improve data quality for modeling

**Tags:** na, missing, clean

**Fields:**
- **df**: The input DataFrame. (DataframeRef)


## ExtractColumn

Convert dataframe column to list.

Use cases:
- Extract data for use in other processing steps
- Prepare column data for plotting or analysis
- Convert categorical data to list for encoding

**Tags:** dataframe, column, list

**Fields:**
- **dataframe**: The input dataframe. (DataframeRef)
- **column_name**: The name of the column to be converted to a list. (str)


## FillNA

Fill missing values in dataframe.

Use cases:
- Handle missing data
- Prepare data for analysis
- Improve data quality

**Tags:** fillna, missing, impute

**Fields:**
- **dataframe**: The DataFrame with missing values. (DataframeRef)
- **value**: Value to use for filling missing values. (Any)
- **method**: Method for filling: value, forward, backward, mean, median (str)
- **columns**: Comma-separated column names to fill. Leave empty for all columns. (str)


## Filter

Filter dataframe based on condition.

Example conditions:
age > 30
age > 30 and salary < 50000
name == 'John Doe'
100 <= price <= 200
status in ['Active', 'Pending']
not (age < 18)

Use cases:
- Extract subset of data meeting specific criteria
- Remove outliers or invalid data points
- Focus analysis on relevant data segments

**Tags:** filter, query, condition

**Fields:**
- **df**: The DataFrame to filter. (DataframeRef)
- **condition**: The filtering condition to be applied to the DataFrame, e.g. column_name > 5. (str)


## FilterNone

Filters out None values from a stream.

Use cases:
- Clean data by removing null values
- Get only valid entries
- Remove placeholder values

**Tags:** filter, none, null, stream

**Fields:**
- **value**: Input stream (Any)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.data.FilterNone.OutputType, NoneType]


## FindRow

Find the first row in a dataframe that matches a given condition.

Example conditions:
age > 30
age > 30 and salary < 50000
name == 'John Doe'
100 <= price <= 200
status in ['Active', 'Pending']
not (age < 18)

Use cases:
- Retrieve specific record based on criteria
- Find first occurrence of a particular condition
- Extract single data point for further analysis

**Tags:** filter, query, condition, single row

**Fields:**
- **df**: The DataFrame to search. (DataframeRef)
- **condition**: The condition to filter the DataFrame, e.g. 'column_name == value'. (str)


## FromList

Convert list of dicts to dataframe.

Use cases:
- Transform list data into structured dataframe
- Prepare list data for analysis or visualization
- Convert API responses to dataframe format

**Tags:** list, dataframe, convert

**Fields:**
- **values**: List of values to be converted, each value will be a row. (list[typing.Any])


## ImportCSV

Convert CSV string to dataframe.

Use cases:
- Import CSV data from string input
- Convert CSV responses from APIs to dataframe

**Tags:** csv, dataframe, import

**Fields:**
- **csv_data**: String input of CSV formatted text. (str)


## JSONToDataframe

Transforms a JSON string into a pandas DataFrame.

Use cases:
- Converting API responses to tabular format
- Preparing JSON data for analysis or visualization
- Structuring unstructured JSON data for further processing

**Tags:** json, dataframe, conversion

**Fields:**
- **text** (str)


## Join

Join two dataframes on specified column.

Use cases:
- Combine data from related tables
- Enrich dataset with additional information
- Link data based on common identifiers

**Tags:** join, merge, column

**Fields:**
- **dataframe_a**: First DataFrame to be merged. (DataframeRef)
- **dataframe_b**: Second DataFrame to be merged. (DataframeRef)
- **join_on**: The column name on which to join the two dataframes. (str)


## LoadCSVAssets

Load dataframes from an asset folder.

Use cases:
- Load multiple dataframes from a folder
- Process multiple datasets in sequence
- Batch import of data files

**Tags:** load, dataframe, file, import

**Fields:**
- **folder**: The asset folder to load the dataframes from. (FolderRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.data.LoadCSVAssets.OutputType, NoneType]


## LoadCSVFile

Load CSV file from file path.

**Tags:** csv, dataframe, import

**Fields:**
- **file_path**: The path to the CSV file to load. (str)


## LoadCSVURL

Load CSV file from URL.

**Tags:** csv, dataframe, import

**Fields:**
- **url**: The URL of the CSV file to load. (str)


## Merge

Merge two dataframes along columns.

Use cases:
- Combine data from multiple sources
- Add new features to existing dataframe
- Merge time series data from different periods

**Tags:** merge, concat, columns

**Fields:**
- **dataframe_a**: First DataFrame to be merged. (DataframeRef)
- **dataframe_b**: Second DataFrame to be merged. (DataframeRef)


## Pivot

Pivot dataframe to reshape data.

Use cases:
- Transform long data to wide format
- Create cross-tabulation tables
- Reorganize data for visualization

**Tags:** pivot, reshape, transform

**Fields:**
- **dataframe**: The DataFrame to pivot. (DataframeRef)
- **index**: Column name to use as index (rows). (str)
- **columns**: Column name to use as columns. (str)
- **values**: Column name to use as values. (str)
- **aggfunc**: Aggregation function: sum, mean, count, min, max, first, last (str)


## Rename

Rename columns in dataframe.

Use cases:
- Standardize column names
- Make column names more descriptive
- Prepare data for specific requirements

**Tags:** rename, columns, names

**Fields:**
- **dataframe**: The DataFrame to rename columns. (DataframeRef)
- **rename_map**: Column rename mapping in format: old1:new1,old2:new2 (str)


## RowIterator

Iterate over rows of a dataframe.

**Fields:**
- **dataframe**: The input dataframe. (DataframeRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.data.RowIterator.OutputType, NoneType]


## SaveCSVDataframeFile

Write a pandas DataFrame to a CSV file.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** files, csv, write, output, save, file

**Fields:**
- **dataframe**: DataFrame to write to CSV (DataframeRef)
- **folder**: Folder where the file will be saved (str)
- **filename**: Name of the CSV file to save. Supports strftime format codes. (str)


## SaveDataframe

Save dataframe in specified folder.

Use cases:
- Export processed data for external use
- Create backups of dataframes

**Tags:** csv, folder, save

**Fields:**
- **df** (DataframeRef)
- **folder**: Name of the output folder. (FolderRef)
- **name**: 
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**


## Schema

Define a schema for a dataframe.

**Tags:** schema, dataframe, create

**Fields:**
- **columns**: The columns to use in the dataframe. (RecordType)


## SelectColumn

Select specific columns from dataframe.

Use cases:
- Extract relevant features for analysis
- Reduce dataframe size by removing unnecessary columns
- Prepare data for specific visualizations or models

**Tags:** dataframe, columns, filter

**Fields:**
- **dataframe**: a dataframe from which columns are to be selected (DataframeRef)
- **columns**: comma separated list of column names (str)


## Slice

Slice a dataframe by rows using start and end indices.

Use cases:
- Extract a specific range of rows from a large dataset
- Create training and testing subsets for machine learning
- Analyze data in smaller chunks

**Tags:** slice, subset, rows

**Fields:**
- **dataframe**: The input dataframe to be sliced. (DataframeRef)
- **start_index**: The starting index of the slice (inclusive). (int)
- **end_index**: The ending index of the slice (exclusive). Use -1 for the last row. (int)


## SortByColumn

Sort dataframe by specified column.

Use cases:
- Arrange data in ascending or descending order
- Identify top or bottom values in dataset
- Prepare data for rank-based analysis

**Tags:** sort, order, column

**Fields:**
- **df** (DataframeRef)
- **column**: The column to sort the DataFrame by. (str)


## ToList

Convert dataframe to list of dictionaries.

Use cases:
- Convert dataframe data for API consumption
- Transform data for JSON serialization
- Prepare data for document-based storage

**Tags:** dataframe, list, convert

**Fields:**
- **dataframe**: The input dataframe to convert. (DataframeRef)


## Collect

Collect items until the end of the stream and return them as a list.

Use cases:
- Gather results from multiple processing steps
- Collect streaming data into batches
- Aggregate outputs from parallel operations

**Tags:** collector, aggregate, list, stream

**Fields:**
- **input_item**: The input item to collect. (Any)

### run

**Args:**
- **context (Any)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## ForEach

Iterate over a list and emit each item sequentially.

Use cases:
- Process each item of a collection in order
- Drive downstream nodes with individual elements

**Tags:** iterator, loop, list, sequence

**Fields:**
- **input_list**: The list of items to iterate over. (list[typing.Any])

### gen_process

**Args:**
- **context (Any)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.control.ForEach.OutputType, NoneType]


## If

Conditionally executes one of two branches based on a condition.

Use cases:
- Branch workflow based on conditions
- Handle different cases in data processing
- Implement decision logic

**Tags:** control, flow, condition, logic, else, true, false, switch, toggle, flow-control

**Fields:**
- **condition**: The condition to evaluate (bool)
- **value**: The value to pass to the next node (Any)

### gen_process

**Args:**
- **context (Any)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.control.If.OutputType, NoneType]


## Reroute

Pass data through unchanged for tidier workflow layouts.

Use cases:
- Organize complex workflows by routing connections
- Create cleaner visual layouts
- Redirect data flow without modification

**Tags:** reroute, passthrough, organize, tidy, flow, connection, redirect

**Fields:**
- **input_value**: Value to pass through unchanged (Any)

### run

**Args:**
- **context (Any)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**


## AddAudio

Add an audio track to a video, replacing or mixing with existing audio.

Use cases:
1. Add background music or narration to a silent video
2. Replace original audio with a new soundtrack
3. Mix new audio with existing video sound

**Tags:** video, audio, soundtrack, merge

**Fields:**
- **video**: The input video to add audio to. (VideoRef)
- **audio**: The audio file to add to the video. (AudioRef)
- **volume**: Volume adjustment for the added audio. 1.0 is original volume. (float)
- **mix**: If True, mix new audio with existing. If False, replace existing audio. (bool)


## AddSubtitles

Add subtitles to a video.

Use cases:
1. Add translations or closed captions to videos
2. Include explanatory text or commentary in educational videos
3. Create lyric videos for music content

**Tags:** video, subtitles, text, caption

**Fields:**
- **video**: The input video to add subtitles to. (VideoRef)
- **chunks**: Audio chunks to add as subtitles. (list[nodetool.metadata.types.AudioChunk])
- **font**: The font to use. (FontRef)
- **align**: Vertical alignment of subtitles. (SubtitleTextAlignment)
- **font_size**: The font size. (int)
- **font_color**: The font color. (ColorRef)


## Blur

Apply a blur effect to a video.

Use cases:
1. Create a dreamy or soft focus effect
2. Obscure or censor specific areas of the video
3. Reduce noise or grain in low-quality footage

**Tags:** video, blur, smooth, soften

**Fields:**
- **video**: The input video to apply blur effect. (VideoRef)
- **strength**: The strength of the blur effect. Higher values create a stronger blur. (float)


## ChromaKey

Apply chroma key (green screen) effect to a video.

Use cases:
1. Remove green or blue background from video footage
2. Create special effects by compositing video onto new backgrounds
3. Produce professional-looking videos for presentations or marketing

**Tags:** video, chroma key, green screen, compositing

**Fields:**
- **video**: The input video to apply chroma key effect. (VideoRef)
- **key_color**: The color to key out (e.g., '#00FF00' for green). (ColorRef)
- **similarity**: Similarity threshold for the key color. (float)
- **blend**: Blending of the keyed area edges. (float)


## ColorBalance

Adjust the color balance of a video.

Use cases:
1. Correct color casts in video footage
2. Enhance specific color tones for artistic effect
3. Normalize color balance across multiple video clips

**Tags:** video, color, balance, adjustment

**Fields:**
- **video**: The input video to adjust color balance. (VideoRef)
- **red_adjust**: Red channel adjustment factor. (float)
- **green_adjust**: Green channel adjustment factor. (float)
- **blue_adjust**: Blue channel adjustment factor. (float)


## Concat

Concatenate multiple video files into a single video, including audio when available.

Use cases:
- Merge multiple video clips into one continuous video
- Combine intro, main content, and outro sequences
- Join video segments from different sources
- Create video compilations and montages

**Tags:** video, concat, merge, combine, audio, +

**Fields:**
- **video_a**: The first video to concatenate. (VideoRef)
- **video_b**: The second video to concatenate. (VideoRef)


## Denoise

Apply noise reduction to a video.

Use cases:
1. Improve video quality by reducing unwanted noise
2. Enhance low-light footage
3. Prepare video for further processing or compression

**Tags:** video, denoise, clean, enhance

**Fields:**
- **video**: The input video to denoise. (VideoRef)
- **strength**: Strength of the denoising effect. Higher values mean more denoising. (float)


## ExtractAudio

Separate and extract audio track from a video file.

Use cases:
- Extract audio for podcasts or music
- Create audio-only versions of video content
- Analyze or transcribe video audio separately
- Reuse audio in different contexts
- Convert video soundtracks to audio files

**Tags:** video, audio, extract, separate, split

**Fields:**
- **video**: The input video to separate. (VideoRef)


## Fps

Get the frames per second (FPS) of a video file.

Use cases:
1. Analyze video properties for quality assessment
2. Determine appropriate playback speed for video editing
3. Ensure compatibility with target display systems

**Tags:** video, analysis, frames, fps

**Fields:**
- **video**: The input video to analyze for FPS. (VideoRef)


## FrameIterator

Extract frames from a video file using OpenCV.

Use cases:
1. Generate image sequences for further processing
2. Extract specific frame ranges from a video
3. Create thumbnails or previews from video content

**Tags:** video, frames, extract, sequence

**Fields:**
- **video**: The input video to extract frames from. (VideoRef)
- **start**: The frame to start extracting from. (int)
- **end**: The frame to stop extracting from. (int)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.video.FrameIterator.OutputType, NoneType]

### get_fps

**Args:**
- **context (ProcessingContext)**

**Returns:** float


## FrameToVideo

Combine a sequence of frames into a single video file.

Use cases:
1. Create time-lapse videos from image sequences
2. Compile processed frames back into a video
3. Generate animations from individual images

**Tags:** video, frames, combine, sequence

**Fields:**
- **frame**: Collect input frames (ImageRef)
- **fps**: The FPS of the output video. (float)

### create_video

**Args:**
- **context (ProcessingContext)**
- **temp_dir (str)**
- **fps (float)**

**Returns:** VideoRef

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**


## ImageToVideo

Generate videos from input images using any supported video provider.
video, image-to-video, i2v, animation, AI, generation, sora, veo

Use cases:
- Animate static images into video sequences
- Create dynamic content from still photographs
- Generate video variations from reference images
- Produce animated visual effects from static artwork
- Convert product photos into engaging video ads

**Tags:** Animates static images into dynamic video content with AI-powered motion.

**Fields:**
- **image**: The input image to animate into a video (ImageRef)
- **model**: The video generation model to use (VideoModel)
- **prompt**: Optional text prompt to guide the video animation (str)
- **negative_prompt**: Text prompt describing what to avoid in the video (str)
- **aspect_ratio**: Aspect ratio for the video (AspectRatio)
- **resolution**: Video resolution (Resolution)
- **num_frames**: Number of frames to generate (provider-specific) (int)
- **guidance_scale**: Classifier-free guidance scale (higher = closer to prompt) (float)
- **num_inference_steps**: Number of denoising steps (int)
- **seed**: Random seed for reproducibility (-1 for random) (int)


## LoadVideoAssets

Load video files from an asset folder.
video, assets, load

Use cases:
- Provide videos for batch processing
- Iterate over stored video assets
- Prepare clips for editing or analysis

**Tags:** 

**Fields:**
- **folder**: The asset folder to load the video files from. (FolderRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.video.LoadVideoAssets.OutputType, NoneType]

### required_inputs

**Args:**


## LoadVideoFile

Read a video file from disk.

Use cases:
- Load videos for processing
- Import video files for editing
- Read video assets for a workflow

**Tags:** video, input, load, file

**Fields:**
- **path**: Path to the video file to read (str)


## Overlay

Overlay one video on top of another, including audio overlay when available.

Use cases:
- Create picture-in-picture effects for commentary videos
- Add watermarks or logos to videos
- Combine multiple video streams
- Create split-screen or multi-view presentations
- Layer video effects over main content

**Tags:** video, overlay, composite, picture-in-picture, audio

**Fields:**
- **main_video**: The main (background) video. (VideoRef)
- **overlay_video**: The video to overlay on top. (VideoRef)
- **x**: X-coordinate for overlay placement. (int)
- **y**: Y-coordinate for overlay placement. (int)
- **scale**: Scale factor for the overlay video. (float)
- **overlay_audio_volume**: Volume of the overlay audio relative to the main audio. (float)


## ResizeNode

Resize a video to a specific width and height.

Use cases:
1. Adjust video resolution for different display requirements
2. Reduce file size by downscaling video
3. Prepare videos for specific platforms with size constraints

**Tags:** video, resize, scale, dimensions

**Fields:**
- **video**: The input video to resize. (VideoRef)
- **width**: The target width. Use -1 to maintain aspect ratio. (int)
- **height**: The target height. Use -1 to maintain aspect ratio. (int)


## Reverse

Reverse the playback of a video.

Use cases:
1. Create artistic effects by playing video in reverse
2. Analyze motion or events in reverse order
3. Generate unique transitions or intros for video projects

**Tags:** video, reverse, backwards, effect

**Fields:**
- **video**: The input video to reverse. (VideoRef)


## Rotate

Rotate a video by a specified angle.

Use cases:
1. Correct orientation of videos taken with a rotated camera
2. Create artistic effects by rotating video content
3. Adjust video for different display orientations

**Tags:** video, rotate, orientation, transform

**Fields:**
- **video**: The input video to rotate. (VideoRef)
- **angle**: The angle of rotation in degrees. (float)


## Saturation

Adjust the color saturation of a video.

Use cases:
1. Enhance color vibrancy in dull or flat-looking footage
2. Create stylistic effects by over-saturating or desaturating video
3. Correct oversaturated footage from certain cameras

**Tags:** video, saturation, color, enhance

**Fields:**
- **video**: The input video to adjust saturation. (VideoRef)
- **saturation**: Saturation level. 1.0 is original, <1 decreases saturation, >1 increases saturation. (float)


## SaveVideo

Save a video to an asset folder.

Use cases:
1. Export processed video to a specific asset folder
2. Save video with a custom name
3. Create a copy of a video in a different location

**Tags:** video, save, file, output

**Fields:**
- **video**: The video to save. (VideoRef)
- **folder**: The asset folder to save the video in. (FolderRef)
- **name**: 
        Name of the output video.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**


## SaveVideoFile

Write a video file to disk.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** video, output, save, file

**Fields:**
- **video**: The video to save (VideoRef)
- **folder**: Folder where the file will be saved (str)
- **filename**: 
        Name of the file to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## SetSpeed

Adjust the playback speed of a video.

Use cases:
1. Create slow-motion effects by decreasing video speed
2. Generate time-lapse videos by increasing playback speed
3. Synchronize video duration with audio or other timing requirements

**Tags:** video, speed, tempo, time

**Fields:**
- **video**: The input video to adjust speed. (VideoRef)
- **speed_factor**: The speed adjustment factor. Values > 1 speed up, < 1 slow down. (float)


## Sharpness

Adjust the sharpness of a video.

Use cases:
1. Enhance detail in slightly out-of-focus footage
2. Correct softness introduced by video compression
3. Create stylistic effects by over-sharpening

**Tags:** video, sharpen, enhance, detail

**Fields:**
- **video**: The input video to sharpen. (VideoRef)
- **luma_amount**: Amount of sharpening to apply to luma (brightness) channel. (float)
- **chroma_amount**: Amount of sharpening to apply to chroma (color) channels. (float)


## Stabilize

Apply video stabilization to reduce camera shake and jitter.

Use cases:
1. Improve quality of handheld or action camera footage
2. Smooth out panning and tracking shots
3. Enhance viewer experience by reducing motion sickness

**Tags:** video, stabilize, smooth, shake-reduction

**Fields:**
- **video**: The input video to stabilize. (VideoRef)
- **smoothing**: Smoothing strength. Higher values result in smoother but potentially more cropped video. (float)
- **crop_black**: Whether to crop black borders that may appear after stabilization. (bool)


## TextToVideo

Generate videos from text prompts using any supported video provider.
video, generation, AI, text-to-video, t2v

Use cases:
- Create videos from text descriptions
- Generate video content from prompts
- Produce short video clips with AI
- Switch between providers without changing workflows

**Tags:** Automatically routes to the appropriate backend (Gemini Veo, HuggingFace).

**Fields:**
- **model**: The video generation model to use (VideoModel)
- **prompt**: Text prompt describing the desired video (str)
- **negative_prompt**: Text prompt describing what to avoid in the video (str)
- **aspect_ratio**: Aspect ratio for the video (AspectRatio)
- **resolution**: Video resolution (Resolution)
- **num_frames**: Number of frames to generate (provider-specific) (int)
- **guidance_scale**: Classifier-free guidance scale (higher = closer to prompt) (float)
- **num_inference_steps**: Number of denoising steps (int)
- **seed**: Random seed for reproducibility (-1 for random) (int)


## Transition

Create a transition effect between two videos, including audio transition when available.

Use cases:
1. Create smooth transitions between video clips in a montage
2. Add professional-looking effects to video projects
3. Blend scenes together for creative storytelling
4. Smoothly transition between audio tracks of different video clips

**Tags:** video, transition, effect, merge, audio

**Fields:**
- **video_a**: The first video in the transition. (VideoRef)
- **video_b**: The second video in the transition. (VideoRef)
- **transition_type**: Type of transition effect (TransitionType)
- **duration**: Duration of the transition effect in seconds. (float)


## Trim

Trim a video to a specific start and end time.

Use cases:
1. Extract specific segments from a longer video
2. Remove unwanted parts from the beginning or end of a video
3. Create shorter clips from a full-length video

**Tags:** video, trim, cut, segment

**Fields:**
- **video**: The input video to trim. (VideoRef)
- **start_time**: The start time in seconds for the trimmed video. (float)
- **end_time**: The end time in seconds for the trimmed video. Use -1 for the end of the video. (float)


## BatchToList

Convert an image batch to a list of image references.

Use cases:
- Convert comfy batch outputs to list format

**Tags:** batch, list, images, processing

**Fields:**
- **batch**: The batch of images to convert. (ImageRef)


## Crop

Crop an image to specified coordinates.

- Remove unwanted borders from images
- Focus on particular subjects within an image
- Simplify images by removing distractions

**Tags:** image, crop

**Fields:**
- **image**: The image to crop. (ImageRef)
- **left**: The left coordinate. (int)
- **top**: The top coordinate. (int)
- **right**: The right coordinate. (int)
- **bottom**: The bottom coordinate. (int)


## Fit

Resize an image to fit within specified dimensions while preserving aspect ratio.

- Resize images for online publishing requirements
- Preprocess images to uniform sizes for machine learning
- Control image display sizes for web development

**Tags:** image, resize, fit

**Fields:**
- **image**: The image to fit. (ImageRef)
- **width**: Width to fit to. (int)
- **height**: Height to fit to. (int)


## GetMetadata

Get metadata about the input image.

Use cases:
- Use width and height for layout calculations
- Analyze image properties for processing decisions
- Gather information for image cataloging or organization

**Tags:** metadata, properties, analysis, information

**Fields:**
- **image**: The input image. (ImageRef)


## ImageToImage

Transform images using text prompts with any supported image provider.
image, transformation, AI, image-to-image, i2i

Use cases:
- Modify existing images with text instructions
- Style transfer and artistic modifications
- Image enhancement and refinement
- Creative image edits guided by prompts

**Tags:** Automatically routes to the appropriate backend (HuggingFace, FAL, MLX).

**Fields:**
- **model**: The image generation model to use (ImageModel)
- **image**: Input image to transform (ImageRef)
- **prompt**: Text prompt describing the desired transformation (str)
- **negative_prompt**: Text prompt describing what to avoid (str)
- **strength**: How much to transform the input image (0.0 = no change, 1.0 = maximum change) (float)
- **guidance_scale**: Classifier-free guidance scale (float)
- **num_inference_steps**: Number of denoising steps (int)
- **target_width**: Target width of the output image (int)
- **target_height**: Target height of the output image (int)
- **seed**: Random seed for reproducibility (-1 for random) (int)
- **scheduler**: Scheduler to use (provider-specific) (str)
- **safety_check**: Enable safety checker (bool)


## LoadImageAssets

Load images from an asset folder.

**Tags:** load, image, file, import

**Fields:**
- **folder**: The asset folder to load the images from. (FolderRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.image.LoadImageAssets.OutputType, NoneType]


## LoadImageFile

Read an image file from disk.

Use cases:
- Load images for processing
- Import photos for editing
- Read image assets for a workflow

**Tags:** image, input, load, file

**Fields:**
- **path**: Path to the image file to read (str)


## LoadImageFolder

Load all images from a folder, optionally including subfolders.

Use cases:
- Batch import images for processing
- Build datasets from a directory tree
- Iterate over photo collections

**Tags:** image, load, folder, files

**Fields:**
- **folder**: Folder to scan for images (str)
- **include_subdirectories**: Include images in subfolders (bool)
- **extensions**: Image file extensions to include (list[str])
- **pattern**: Pattern to match image files (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.image.LoadImageFolder.OutputType, NoneType]


## Paste

Paste one image onto another at specified coordinates.

Use cases:
- Add watermarks or logos to images
- Combine multiple image elements
- Create collages or montages

**Tags:** paste, composite, positioning, overlay

**Fields:**
- **image**: The image to paste into. (ImageRef)
- **paste**: The image to paste. (ImageRef)
- **left**: The left coordinate. (int)
- **top**: The top coordinate. (int)


## Resize

Change image dimensions to specified width and height.

- Preprocess images for machine learning model inputs
- Optimize images for faster web page loading
- Create uniform image sizes for layouts

**Tags:** image, resize

**Fields:**
- **image**: The image to resize. (ImageRef)
- **width**: The target width. (int)
- **height**: The target height. (int)


## SaveImage

Save an image to specified asset folder with customizable name format.

Use cases:
- Save generated images with timestamps
- Organize outputs into specific folders
- Create backups of processed images

**Tags:** save, image, folder, naming

**Fields:**
- **image**: The image to save. (ImageRef)
- **folder**: The asset folder to save the image in. (FolderRef)
- **name**: 
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**

### result_for_client

**Args:**
- **result (dict[str, typing.Any])**

**Returns:** dict[str, typing.Any]


## SaveImageFile

Write an image to disk.

Use cases:
- Save processed images
- Export edited photos
- Archive image results

**Tags:** image, output, save, file

**Fields:**
- **image**: The image to save (ImageRef)
- **folder**: Folder where the file will be saved (str)
- **filename**: 
        The name of the image file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)
- **overwrite**: Overwrite the file if it already exists, otherwise file will be renamed (bool)


## Scale

Enlarge or shrink an image by a scale factor.

- Adjust image dimensions for display galleries
- Standardize image sizes for machine learning datasets
- Create thumbnail versions of images

**Tags:** image, resize, scale

**Fields:**
- **image**: The image to scale. (ImageRef)
- **scale**: The scale factor. (float)


## TextToImage

Generate images from text prompts using any supported image provider.
image, generation, AI, text-to-image, t2i

Use cases:
- Create images from text descriptions
- Switch between providers without changing workflows
- Generate images with different AI models
- Cost-optimize by choosing different providers

**Tags:** Automatically routes to the appropriate backend (HuggingFace, FAL, MLX).

**Fields:**
- **model**: The image generation model to use (ImageModel)
- **prompt**: Text prompt describing the desired image (str)
- **negative_prompt**: Text prompt describing what to avoid in the image (str)
- **width**: Width of the generated image (int)
- **height**: Height of the generated image (int)
- **guidance_scale**: Classifier-free guidance scale (higher = closer to prompt) (float)
- **num_inference_steps**: Number of denoising steps (int)
- **seed**: Random seed for reproducibility (-1 for random) (int)
- **safety_check**: Enable safety checker to filter inappropriate content (bool)


## AssetFolderInput

Accepts an asset folder as a parameter for workflows.

**Tags:** input, parameter, folder, path, folderpath, local_folder, filesystem

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The folder to use as input. (FolderRef)
- **description**: The description of the input for the workflow. (str)


## AudioInput

Accepts a reference to an audio asset for workflows, specified by an 'AudioRef'.  An 'AudioRef' points to audio data that can be used for playback, transcription, analysis, or processing by audio-capable models.

Use cases:
- Load an audio file for speech-to-text transcription.
- Analyze sound for specific events or characteristics.
- Provide audio input to models for tasks like voice recognition or music generation.
- Process audio for enhancement or feature extraction.

**Tags:** input, parameter, audio, sound, voice, speech, asset

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The audio to use as input. (AudioRef)
- **description**: The description of the input for the workflow. (str)


## BooleanInput

Accepts a boolean (true/false) value as a parameter for workflows.  This input is used for binary choices, enabling or disabling features, or controlling conditional logic paths.

Use cases:
- Toggle features or settings on or off.
- Set binary flags to control workflow behavior.
- Make conditional choices within a workflow (e.g., proceed if true).

**Tags:** input, parameter, boolean, bool, toggle, switch, flag

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (bool)
- **description**: The description of the input for the workflow. (str)


## ColorInput

Accepts a color value as a parameter for workflows.

**Tags:** input, parameter, color, color_picker, color_input

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The color to use as input. (ColorRef)
- **description**: The description of the input for the workflow. (str)


## DataframeInput

Accepts a reference to a dataframe asset for workflows.

**Tags:** input, parameter, dataframe, table, data

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The dataframe to use as input. (DataframeRef)
- **description**: The description of the input for the workflow. (str)


## DocumentFileInput

Accepts a local file path pointing to a document and converts it into a 'DocumentRef'.

Use cases:
- Directly load a document (e.g., PDF, TXT, DOCX) from a specified local file path.
- Convert a local file path into a 'DocumentRef' that can be consumed by other document-processing nodes.
- Useful for development or workflows that have legitimate access to the local filesystem.
- To provide an existing 'DocumentRef', use 'DocumentInput'.

**Tags:** input, parameter, document, file, path, local_file, load

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The path to the document file. (str)
- **description**: The description of the input for the workflow. (str)


## DocumentInput

Accepts a reference to a document asset for workflows, specified by a 'DocumentRef'.  A 'DocumentRef' points to a structured document (e.g., PDF, DOCX, TXT) which can be processed or analyzed. This node is used when the workflow needs to operate on a document as a whole entity, potentially including its structure and metadata, rather than just raw text.

Use cases:
- Load a specific document (e.g., PDF, Word, text file) for content extraction or analysis.
- Pass a document to models that are designed to process specific document formats.
- Manage documents as distinct assets within a workflow.
- If you have a local file path and need to convert it to a 'DocumentRef', consider using 'DocumentFileInput'.

**Tags:** input, parameter, document, file, asset, reference

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The document to use as input. (DocumentRef)
- **description**: The description of the input for the workflow. (str)


## FilePathInput

Accepts a local filesystem path (to a file or directory) as input for workflows.

Use cases:
- Provide a local path to a specific file or directory for processing.
- Specify an input or output location on the local filesystem for a development task.
- Load local datasets or configuration files not managed as assets.
- Not available in production: raises an error if used in a production environment.

**Tags:** input, parameter, path, filepath, directory, local_file, filesystem

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The path to use as input. (str)
- **description**: The description of the input for the workflow. (str)


## FloatInput

Accepts a floating-point number as a parameter for workflows, typically constrained by a minimum and maximum value.  This input allows for precise numeric settings, such as adjustments, scores, or any value requiring decimal precision.

Use cases:
- Specify a numeric value within a defined range (e.g., 0.0 to 1.0).
- Set thresholds, confidence scores, or scaling factors.
- Configure continuous parameters like opacity, volume, or temperature.

**Tags:** input, parameter, float, number, decimal, range

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (float)
- **description**: The description of the input for the workflow. (str)
- **min** (float)
- **max** (float)


## FolderPathInput

Accepts a folder path as a parameter for workflows.

**Tags:** input, parameter, folder, path, folderpath, local_folder, filesystem

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The folder path to use as input. (str)
- **description**: The description of the input for the workflow. (str)


## HuggingFaceModelInput

Accepts a Hugging Face model as a parameter for workflows.

**Tags:** input, parameter, model, huggingface, hugging_face, model_name

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The Hugging Face model to use as input. (HuggingFaceModel)
- **description**: The description of the input for the workflow. (str)


## ImageInput

Accepts a reference to an image asset for workflows, specified by an 'ImageRef'.  An 'ImageRef' points to image data that can be used for display, analysis, or processing by vision models.

Use cases:
- Load an image for visual processing or analysis.
- Provide an image as input to computer vision models (e.g., object detection, image classification).
- Select an image for manipulation, enhancement, or inclusion in a document.
- Display an image within a workflow interface.

**Tags:** input, parameter, image, picture, graphic, visual, asset

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The image to use as input. (ImageRef)
- **description**: The description of the input for the workflow. (str)


## ImageModelInput

Accepts an image generation model as a parameter for workflows.

**Tags:** input, parameter, model, image, generation

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The image generation model to use as input. (ImageModel)
- **description**: The description of the input for the workflow. (str)


## IntegerInput

Accepts an integer (whole number) as a parameter for workflows, typically constrained by a minimum and maximum value.  This input is used for discrete numeric values like counts, indices, or iteration limits.

Use cases:
- Specify counts or quantities (e.g., number of items, iterations).
- Set index values for accessing elements in a list or array.
- Configure discrete numeric parameters like age, steps, or quantity.

**Tags:** input, parameter, integer, number, count, index, whole_number

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (int)
- **description**: The description of the input for the workflow. (str)
- **min** (int)
- **max** (int)


## LanguageModelInput

Accepts a language model as a parameter for workflows.

**Tags:** input, parameter, model, language, model_name

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The language model to use as input. (LanguageModel)
- **description**: The description of the input for the workflow. (str)


## RealtimeAudioInput

Accepts streaming audio data for workflows.

**Tags:** input, parameter, audio, sound, voice, speech, asset

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The audio to use as input. (AudioRef)
- **description**: The description of the input for the workflow. (str)


## StringInput

Accepts a string value as a parameter for workflows.

Use cases:
- Define a name for an entity or process.
- Specify a label for a component or output.
- Enter a short keyword or search term.
- Provide a simple configuration value (e.g., an API key, a model name).
- If you need to input multi-line text or the content of a file, use 'DocumentFileInput'.

**Tags:** input, parameter, string, text, label, name, value

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value** (str)
- **description**: The description of the input for the workflow. (str)


## StringListInput

Accepts a list of strings as a parameter for workflows.

**Tags:** input, parameter, string, text, label, name, value

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The list of strings to use as input. (list[str])
- **description**: The description of the input for the workflow. (str)


## VideoInput

Accepts a reference to a video asset for workflows, specified by a 'VideoRef'.  A 'VideoRef' points to video data that can be used for playback, analysis, frame extraction, or processing by video-capable models.

Use cases:
- Load a video file for processing or content analysis.
- Analyze video content for events, objects, or speech.
- Extract frames or audio tracks from a video.
- Provide video input to models that understand video data.

**Tags:** input, parameter, video, movie, clip, visual, asset

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **value**: The video to use as input. (VideoRef)
- **description**: The description of the input for the workflow. (str)


## ExecuteBash

Executes Bash script with safety restrictions.

**Tags:** bash, shell, code, execute

**Fields:**
- **code**: Bash script to execute as-is. Dynamic inputs are provided as env vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'. (str)
- **image**: Docker image to use for execution (BashImage)
- **execution_mode**: Execution mode: 'docker' or 'subprocess' (ExecutionMode)
- **stdin**: String to write to process stdin before any streaming input. Use newlines to separate lines. (str)

### finalize

Stop any running Docker container for this node.


**Args:**

- **context**: Processing context (unused).


**Returns:**

None
**Args:**
- **context (ProcessingContext)**

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## ExecuteCommand

Executes a single shell command inside a Docker container.

IMPORTANT: Only enabled in non-production environments

**Tags:** command, execute, shell, bash, sh

**Fields:**
- **command**: Single command to run via the selected shell. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'. (str)
- **image**: Docker image to use for execution (CommandImage)
- **execution_mode**: Execution mode: 'docker' or 'subprocess' (ExecutionMode)
- **stdin**: String to write to process stdin before any streaming input. Use newlines to separate lines. (str)

### finalize

Stop any running Docker container for this node.


**Args:**

- **context**: Processing context (unused).


**Returns:**

None
**Args:**
- **context (ProcessingContext)**

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## ExecuteJavaScript

Executes JavaScript (Node.js) code with safety restrictions.

**Tags:** javascript, nodejs, code, execute

**Fields:**
- **code**: JavaScript code to execute as-is under Node.js. Dynamic inputs are provided as local vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'. (str)
- **image**: Docker image to use for execution (JavaScriptImage)
- **execution_mode**: Execution mode: 'docker' or 'subprocess' (ExecutionMode)
- **stdin**: String to write to process stdin before any streaming input. Use newlines to separate lines. (str)

### finalize

Stop any running Docker container for this node.


**Args:**

- **context**: Processing context (unused).


**Returns:**

None
**Args:**
- **context (ProcessingContext)**

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## ExecuteLua

Executes Lua code with a local sandbox (no Docker).

**Tags:** lua, code, execute, sandbox

**Fields:**
- **code**: Lua code to execute as-is in a restricted environment. Dynamic inputs are provided as variables. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'. (str)
- **executable**: Lua executable to use (LuaExecutable)
- **execution_mode**: Execution mode: 'docker' or 'subprocess' (ExecutionMode)
- **timeout_seconds**: Max seconds to allow execution before forced stop (int)
- **stdin**: String to write to process stdin before any streaming input. Use newlines to separate lines. (str)

### finalize

**Args:**
- **context (ProcessingContext)**

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## ExecutePython

Executes Python code with safety restrictions.

Use cases:
- Run custom data transformations
- Prototype node functionality
- Debug and testing workflows

IMPORTANT: Only enabled in non-production environments

**Tags:** python, code, execute

**Fields:**
- **code**: Python code to execute as-is. Dynamic inputs are provided as local vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'. (str)
- **image**: Docker image to use for execution (PythonImage)
- **execution_mode**: Execution mode: 'docker' or 'subprocess' (ExecutionMode)
- **stdin**: String to write to process stdin before any streaming input. Use newlines to separate lines. (str)

### finalize

Stop any running Docker container for this node.


**Args:**

- **context**: Processing context (unused).


**Returns:**

None
**Args:**
- **context (ProcessingContext)**

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## ExecuteRuby

Executes Ruby code with safety restrictions.

**Tags:** ruby, code, execute

**Fields:**
- **code**: Ruby code to execute as-is. Dynamic inputs are provided as env vars. Stdout lines are emitted on 'stdout'; stderr lines on 'stderr'. (str)
- **image**: Docker image to use for execution (RubyImage)
- **execution_mode**: Execution mode: 'docker' or 'subprocess' (ExecutionMode)
- **stdin**: String to write to process stdin before any streaming input. Use newlines to separate lines. (str)

### finalize

Stop any running Docker container for this node.


**Args:**

- **context**: Processing context (unused).


**Returns:**

None
**Args:**
- **context (ProcessingContext)**

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## AutomaticSpeechRecognition

Transcribe audio to text using automatic speech recognition models.

Use cases:
- Transcribe recorded audio to text
- Generate subtitles from video audio
- Convert voice notes to written text
- Process meeting recordings
- Enable voice-based data entry

**Tags:** audio, speech, recognition, transcription, ASR, whisper

**Fields:**
- **model** (ASRModel)
- **audio**: The audio to transcribe (AudioRef)


## CapitalizeText

Capitalizes only the first character.

Use cases:
- Formatting short labels or sentences
- Cleaning up LLM output before UI rendering
- Quickly fixing lowercase starts after concatenation

**Tags:** text, transform, capitalize, format

**Fields:**
- **text** (str)


## Chunk

Splits text into chunks of specified word length.

Use cases:
- Preparing text for processing by models with input length limits
- Creating manageable text segments for parallel processing
- Generating summaries of text sections

**Tags:** text, chunk, split

**Fields:**
- **text** (str)
- **length** (int)
- **overlap** (int)
- **separator** (str)


## CollapseWhitespace

Collapses consecutive whitespace into single separators.

Use cases:
- Normalizing pasted text from PDFs or chat logs
- Cleaning prompts with erratic spacing
- Converting multi-line input into succinct sentences

**Tags:** text, whitespace, normalize, clean, remove

**Fields:**
- **text** (str)
- **preserve_newlines**: Keep newline characters instead of replacing them (bool)
- **replacement**: String used to replace whitespace runs (str)
- **trim_edges**: Strip whitespace before collapsing (bool)


## Collect

Collects a stream of text inputs into a single concatenated string.

Use cases:
- Combine multiple streaming text outputs
- Accumulate results from iterative processes
- Build composite text from multiple sources
- Aggregate log messages or status updates

**Tags:** text, collect, list, stream, aggregate

**Fields:**
- **input_item** (str)
- **separator** (str)

### run

**Args:**
- **context (ProcessingContext)**
- **inputs (NodeInputs)**
- **outputs (NodeOutputs)**

**Returns:** None


## Compare

Compares two text values and reports ordering.

Use cases:
- Checking if two strings are identical before branching
- Determining lexical order for sorting or deduplication
- Normalizing casing/spacing before compares

**Tags:** text, compare, equality, sort, equals, =

**Fields:**
- **text_a** (str)
- **text_b** (str)
- **case_sensitive**: Compare without lowercasing (bool)
- **trim_whitespace**: Strip leading/trailing whitespace before comparing (bool)


## Concat

Concatenates two text inputs into a single output.

Use cases:
- Joining outputs from multiple text processing nodes
- Combining parts of sentences or paragraphs
- Merging text data from different sources

**Tags:** text, concatenation, combine, +

**Fields:**
- **a** (str)
- **b** (str)


## Contains

Checks if text contains a specified substring.

Use cases:
- Ensuring safety or guard phrases appear
- Rejecting inputs when banned terms exist
- Matching multiple keywords with any/all logic

**Tags:** text, compare, validate, substring, string

**Fields:**
- **text** (str)
- **substring** (str)
- **search_values**: Optional list of additional substrings to check (list[str])
- **case_sensitive** (bool)
- **match_mode**: ANY requires one match, ALL needs every value, NONE ensures none (MatchMode)


## CountTokens

Counts the number of tokens in text using tiktoken.

Use cases:
- Checking text length for LLM input limits
- Estimating API costs
- Managing token budgets in text processing

**Tags:** text, tokens, count, encoding

**Fields:**
- **text** (str)
- **encoding**: The tiktoken encoding to use for token counting (TiktokenEncoding)


## EndsWith

Checks if text ends with a specified suffix.

Use cases:
- Validating file extensions
- Checking string endings
- Filtering text based on ending content

**Tags:** text, check, suffix, compare, validate, substring, string

**Fields:**
- **text** (str)
- **suffix** (str)


## Equals

Checks if two text inputs are equal.

Use cases:
- Branching workflows when user input matches an expected value
- Guarding against duplicates before saving assets
- Quickly comparing normalized prompts or identifiers

**Tags:** text, compare, equals, match, =

**Fields:**
- **text_a** (str)
- **text_b** (str)
- **case_sensitive**: Disable lowercasing before compare (bool)
- **trim_whitespace**: Strip leading/trailing whitespace prior to comparison (bool)


## Extract

Extracts a substring from input text.

Use cases:
- Extracting specific portions of text for analysis
- Trimming unwanted parts from text data
- Focusing on relevant sections of longer documents

**Tags:** text, extract, substring

**Fields:**
- **text** (str)
- **start** (int)
- **end** (int)


## ExtractJSON

Extracts data from JSON using JSONPath expressions.

Use cases:
- Retrieving specific fields from complex JSON structures
- Filtering and transforming JSON data for analysis
- Extracting nested data from API responses or configurations

**Tags:** json, extract, jsonpath

**Fields:**
- **text** (str)
- **json_path** (str)
- **find_all** (bool)


## ExtractRegex

Extracts substrings matching regex groups from text.

Use cases:
- Extracting structured data (e.g., dates, emails) from unstructured text
- Parsing specific patterns in log files or documents
- Isolating relevant information from complex text formats

**Tags:** text, regex, extract

**Fields:**
- **text** (str)
- **regex** (str)
- **dotall** (bool)
- **ignorecase** (bool)
- **multiline** (bool)


## FilterRegexString

Filters a stream of strings using regular expressions.

Use cases:
- Filter strings using complex patterns
- Extract strings matching specific formats (emails, dates, etc.)

**Tags:** filter, regex, pattern, text, stream

**Fields:**
- **value**: Input string stream (str)
- **pattern**: The regular expression pattern to match against. (str)
- **full_match**: Whether to match the entire string or find pattern anywhere in string (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.text.FilterRegexString.OutputType, NoneType]


## FilterString

Filters a stream of strings based on various criteria.

Use cases:
- Filter strings by length
- Filter strings containing specific text
- Filter strings by prefix/suffix

**Tags:** filter, strings, text, stream

**Fields:**
- **value**: Input string stream (str)
- **filter_type**: The type of filter to apply (FilterType)
- **criteria**: The filtering criteria (text to match or length as string) (str)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.text.FilterString.OutputType, NoneType]


## FindAllRegex

Finds all regex matches in text as separate substrings.

Use cases:
- Identifying all occurrences of a pattern in text
- Extracting multiple instances of structured data
- Analyzing frequency and distribution of specific text patterns

**Tags:** text, regex, find

**Fields:**
- **text** (str)
- **regex** (str)
- **dotall** (bool)
- **ignorecase** (bool)
- **multiline** (bool)


## FormatText

Replaces placeholders in a string with dynamic inputs using Jinja2 templating.

This node is dynamic and can be used to format text with dynamic properties.

Use cases:
- Generating personalized messages with dynamic content
- Creating parameterized queries or commands
- Formatting and filtering text output based on variable inputs

Examples:
- text: "Hello, {{ name }}!"
- text: "Title: {{ title|truncate(20) }}"
- text: "Name: {{ name|upper }}"

Available filters:
- truncate(length): Truncates text to given length
- upper: Converts text to uppercase
- lower: Converts text to lowercase
- title: Converts text to title case
- trim: Removes whitespace from start/end
- replace(old, new): Replaces substring
- default(value): Sets default if value is undefined
- first: Gets first character/item
- last: Gets last character/item
- length: Gets length of string/list
- sort: Sorts list
- join(delimiter): Joins list with delimiter

**Tags:** text, template, formatting

**Fields:**
- **template**: 
    Examples:
    - text: "Hello, {{ name }}!"
    - text: "Title: {{ title|truncate(20) }}"
    - text: "Name: {{ name|upper }}" 

    Available filters:
    - truncate(length): Truncates text to given length
    - upper: Converts text to uppercase
    - lower: Converts text to lowercase
    - title: Converts text to title case
    - trim: Removes whitespace from start/end
    - replace(old, new): Replaces substring
    - default(value): Sets default if value is undefined
    - first: Gets first character/item
    - last: Gets last character/item
    - length: Gets length of string/list
    - sort: Sorts list
    - join(delimiter): Joins list with delimiter
 (str)


## HasLength

Checks if text length meets specified conditions.

Use cases:
- Validating input length requirements
- Filtering text by length
- Checking content size constraints

**Tags:** text, check, length, compare, validate, whitespace, string

**Fields:**
- **text** (str)
- **min_length** (int)
- **max_length** (int)
- **exact_length** (int)


## HtmlToText

Converts HTML content to plain text using html2text.

Use cases:
- Converting HTML documents to readable plain text
- Extracting text content from web pages
- Cleaning HTML markup from text data
- Processing HTML emails or documents

**Tags:** html, convert, text, parse, extract

**Fields:**
- **html**: HTML content to convert (str)
- **base_url**: Base URL for resolving relative links (str)
- **body_width**: Width for text wrapping (int)
- **ignore_images**: Whether to ignore image tags (bool)
- **ignore_mailto_links**: Whether to ignore mailto links (bool)


## IndexOf

Finds the position of a substring in text.

Use cases:
- Locating markers to drive downstream slices
- Building quick validations before parsing
- Detecting repeated terms by scanning from the end

**Tags:** text, search, find, substring

**Fields:**
- **text** (str)
- **substring** (str)
- **case_sensitive** (bool)
- **start_index**: Index to begin the search from (int)
- **end_index**: Optional exclusive end index for the search (int)
- **search_from_end**: Use the last occurrence instead of the first (bool)


## IsEmpty

Checks if text is empty or contains only whitespace.

Use cases:
- Validating required text fields
- Filtering out empty content
- Checking for meaningful input

**Tags:** text, check, empty, compare, validate, whitespace, string

**Fields:**
- **text** (str)
- **trim_whitespace** (bool)


## Join

Joins a list of strings into a single string using a specified separator.

Use cases:
- Combining multiple text elements with a consistent delimiter
- Creating comma-separated lists from individual items
- Assembling formatted text from array elements

**Tags:** text, join, combine, +, add, concatenate

**Fields:**
- **strings** (list[typing.Any])
- **separator** (str)


## Length

Measures text length as characters, words, or lines.

Use cases:
- Quickly gating prompts by size before LLM calls
- Showing word or line counts in mini apps
- Tracking character budgets for UI copy

**Tags:** text, analyze, length, count

**Fields:**
- **text** (str)
- **measure**: Choose whether to count characters, words, or lines (Measure)
- **trim_whitespace**: Strip whitespace before counting (bool)


## LoadTextAssets

Load text files from an asset folder.

Use cases:
- Loading multiple text files for batch processing
- Importing text content from a directory
- Processing collections of text documents

**Tags:** load, text, file, import

**Fields:**
- **folder**: The asset folder to load the text files from. (FolderRef)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.text.LoadTextAssets.OutputType, NoneType]


## PadText

Pads text to a target length.

Use cases:
- Aligning tabular text outputs
- Creating fixed-width fields for legacy systems
- Left-padding numbers with zeros

**Tags:** text, pad, length, format

**Fields:**
- **text** (str)
- **length** (int)
- **pad_character**: Single character to use for padding (str)
- **direction**: Where padding should be applied (PadDirection)


## ParseJSON

Parses a JSON string into a Python object.

Use cases:
- Converting JSON API responses for further processing
- Preparing structured data for analysis or storage
- Extracting configuration or settings from JSON files

**Tags:** json, parse, convert

**Fields:**
- **text** (str)


## RegexMatch

Find all matches of a regex pattern in text.

Use cases:
- Extract specific patterns from text
- Validate text against patterns
- Find all occurrences of a pattern

**Tags:** regex, search, pattern, match

**Fields:**
- **text**: Text to search in (str)
- **pattern**: Regular expression pattern (str)
- **group**: Capture group to extract (0 for full match) (int)


## RegexReplace

Replace text matching a regex pattern.

Use cases:
- Clean or standardize text
- Remove unwanted patterns
- Transform text formats

**Tags:** regex, replace, substitute

**Fields:**
- **text**: Text to perform replacements on (str)
- **pattern**: Regular expression pattern (str)
- **replacement**: Replacement text (str)
- **count**: Maximum replacements (0 for unlimited) (int)


## RegexSplit

Split text using a regex pattern as delimiter.

Use cases:
- Parse structured text
- Extract fields from formatted strings
- Tokenize text

**Tags:** regex, split, tokenize

**Fields:**
- **text**: Text to split (str)
- **pattern**: Regular expression pattern to split on (str)
- **maxsplit**: Maximum number of splits (0 for unlimited) (int)


## RegexValidate

Check if text matches a regex pattern.

Use cases:
- Validate input formats (email, phone, etc)
- Check text structure
- Filter text based on patterns

**Tags:** regex, validate, check

**Fields:**
- **text**: Text to validate (str)
- **pattern**: Regular expression pattern (str)


## RemovePunctuation

Removes punctuation characters from text.

Use cases:
- Cleaning transcripts before keyword search
- Preparing identifiers for filesystem safe names
- Simplifying comparisons by stripping symbols

**Tags:** text, cleanup, punctuation, normalize

**Fields:**
- **text** (str)
- **replacement**: String to insert where punctuation was removed (str)
- **punctuation**: Characters that should be removed or replaced (str)


## Replace

Replaces a substring in a text with another substring.

Use cases:
- Correcting or updating specific text patterns
- Sanitizing or normalizing text data
- Implementing simple text transformations

**Tags:** text, replace, substitute

**Fields:**
- **text** (str)
- **old** (str)
- **new** (str)


## SaveText

Saves input text to a file in the assets folder.

Use cases:
- Persisting processed text results
- Creating text files for downstream nodes or external use
- Archiving text data within the workflow

**Tags:** text, save, file

**Fields:**
- **text** (str)
- **folder**: Name of the output folder. (FolderRef)
- **name**: 
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**


## SaveTextFile

Saves input text to a file in the assets folder.

**Tags:** text, save, file

**Fields:**
- **text** (str)
- **folder**: Path to the output folder. (str)
- **name**: 
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## Slice

Slices text using Python's slice notation (start:stop:step).

Use cases:
- Extracting specific portions of text with flexible indexing
- Reversing text using negative step
- Taking every nth character with step parameter

Examples:
- start=0, stop=5: first 5 characters
- start=-5: last 5 characters
- step=2: every second character
- step=-1: reverse the text

**Tags:** text, slice, substring

**Fields:**
- **text** (str)
- **start** (int)
- **stop** (int)
- **step** (int)


## Slugify

Converts text into a slug suitable for URLs or IDs.

Use cases:
- Generating workflow IDs from titles
- Creating asset filenames from prompts
- Producing URL-safe paths for mini apps

**Tags:** text, slug, normalize, id

**Fields:**
- **text** (str)
- **separator** (str)
- **lowercase** (bool)
- **allow_unicode**: Keep unicode letters instead of converting to ASCII (bool)


## Split

Separates text into a list of strings based on a specified delimiter.

Use cases:
- Parsing CSV or similar delimited data
- Breaking down sentences into words or phrases
- Extracting specific elements from structured text

**Tags:** text, split, tokenize

**Fields:**
- **text** (str)
- **delimiter** (str)


## StartsWith

Checks if text starts with a specified prefix.

Use cases:
- Validating string prefixes
- Filtering text based on starting content
- Checking file name patterns

**Tags:** text, check, prefix, compare, validate, substring, string

**Fields:**
- **text** (str)
- **prefix** (str)


## StripAccents

Removes accent marks while keeping base characters.

Use cases:
- Creating ASCII-only identifiers from user input
- Normalizing prompts that mix accented and plain characters
- Simplifying comparisons against datasets lacking accents

**Tags:** text, cleanup, accents, normalize

**Fields:**
- **text** (str)
- **preserve_non_ascii**: Keep non-ASCII characters that are not accents (bool)


## SurroundWith

Wraps text with the provided prefix and suffix.

Use cases:
- Adding quotes or brackets before exporting values
- Ensuring prompts include guard rails or markup tokens
- Building template strings without using Format nodes

**Tags:** text, format, surround, decorate

**Fields:**
- **text** (str)
- **prefix** (str)
- **suffix** (str)
- **skip_if_wrapped**: Do not add duplicates if the text is already wrapped (bool)


## Template

Uses Jinja2 templating to format strings with variables and filters. This node is dynamic and can be used to format text with dynamic inputs.

Use cases:
- Generating personalized messages with dynamic content
- Creating parameterized queries or commands
- Formatting and filtering text output based on variable inputs

Examples:
- text: "Hello, {{ name }}!"
- text: "Title: {{ title|truncate(20) }}"
- text: "Name: {{ name|upper }}"

Available filters:
- truncate(length): Truncates text to given length
- upper: Converts text to uppercase
- lower: Converts text to lowercase
- title: Converts text to title case
- trim: Removes whitespace from start/end
- replace(old, new): Replaces substring
- default(value): Sets default if value is undefined
- first: Gets first character/item
- last: Gets last character/item
- length: Gets length of string/list
- sort: Sorts list
- join(delimiter): Joins list with delimiter

**Tags:** text, template, formatting, format, combine, concatenate, +, add, variable, replace, filter

**Fields:**
- **string**: 
    Examples:
    - text: "Hello, {{ name }}!"
    - text: "Title: {{ title|truncate(20) }}"
    - text: "Name: {{ name|upper }}"

    Available filters:
    - truncate(length): Truncates text to given length
    - upper: Converts text to uppercase
    - lower: Converts text to lowercase
    - title: Converts text to title case
    - trim: Removes whitespace from start/end
    - replace(old, new): Replaces substring
    - default(value): Sets default if value is undefined
    - first: Gets first character/item
    - last: Gets last character/item
    - length: Gets length of string/list
    - sort: Sorts list
    - join(delimiter): Joins list with delimiter
 (str)
- **values**: 
        The values to replace in the string.
        - If a string, it will be used as the format string.
        - If a list, it will be used as the format arguments.
        - If a dictionary, it will be used as the template variables.
        - If an object, it will be converted to a dictionary using the object's __dict__ method.
         (Any)


## ToLowercase

Converts text to lowercase.

Use cases:
- Preparing data for case-insensitive comparisons
- Generating lowercase filenames or IDs
- Normalizing prompts before hashing

**Tags:** text, transform, lowercase, format

**Fields:**
- **text** (str)


## ToString

Converts any input value to its string representation.

Use cases:
- Convert numbers, objects, or complex types to strings
- Prepare data for text output or logging
- Debug values by viewing their representations
- Standardize data types in text workflows

**Tags:** text, string, convert, repr, str, cast

**Fields:**
- **value** (Any)
- **mode**: Conversion mode: use `str(value)` or `repr(value)`. (Mode)


## ToTitlecase

Converts text to title case.

Use cases:
- Cleaning user provided titles before display
- Normalizing headings in generated documents
- Making list entries easier to scan

**Tags:** text, transform, titlecase, format

**Fields:**
- **text** (str)


## ToUppercase

Converts text to uppercase.

Use cases:
- Normalizing identifiers before comparison
- Preparing titles that must display in all caps
- Converting prompts to a consistent casing convention

**Tags:** text, transform, uppercase, format

**Fields:**
- **text** (str)


## TrimWhitespace

Trims whitespace from the start and/or end of text.

Use cases:
- Cleaning user input before validation
- Removing accidental spaces after concatenation
- Prepping prompts for exact comparisons

**Tags:** text, whitespace, clean, remove

**Fields:**
- **text** (str)
- **trim_start** (bool)
- **trim_end** (bool)


## TruncateText

Truncates text to a maximum length.

Use cases:
- Enforcing LLM input limits before sending prompts
- Creating previews in UI cards
- Guarding downstream systems that expect short strings

**Tags:** text, truncate, length, clip

**Fields:**
- **text** (str)
- **max_length** (int)
- **ellipsis**: Optional suffix appended when truncation occurs (str)


## FileWatchTrigger

Trigger node that monitors filesystem changes.
This trigger uses the watchdog library to monitor a directory or file
for changes. When a change is detected, an event is emitted containing:
- The path of the changed file
- The type of change (created, modified, deleted, moved)
- Timestamp of the event

This trigger is useful for:
- Processing files as they arrive in a directory
- Triggering workflows on configuration changes
- Building file-based automation pipelines

**Tags:** 

**Fields:**
- **max_events**: Maximum number of events to process (0 = unlimited) (int)
- **path**: Path to watch (file or directory) (str)
- **recursive**: Watch subdirectories recursively (bool)
- **patterns**: File patterns to watch (e.g., ['*.txt', '*.json']) (list[str])
- **ignore_patterns**: File patterns to ignore (list[str])
- **events**: Types of events to watch for (list[str])
- **debounce_seconds**: Debounce time to avoid duplicate events (float)

### cleanup_trigger

Stop the filesystem watcher.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### setup_trigger

Start the filesystem watcher.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### wait_for_event

Wait for the next filesystem event.
**Args:**
- **context (ProcessingContext)**

**Returns:** FileWatchTriggerOutput | None


## IntervalTrigger

Trigger node that fires at regular time intervals.
This trigger emits events at a configured interval, similar to a timer
or scheduler. Each event contains:
- The tick number (how many times the trigger has fired)
- The current timestamp
- The configured interval

This trigger is useful for:
- Periodic data collection or polling
- Scheduled batch processing
- Heartbeat or keepalive workflows
- Time-based automation

**Tags:** 

**Fields:**
- **max_events**: Maximum number of events to process (0 = unlimited) (int)
- **interval_seconds**: Interval between triggers in seconds (float)
- **initial_delay_seconds**: Delay before the first trigger fires (float)
- **emit_on_start**: Whether to emit an event immediately on start (bool)
- **include_drift_compensation**: Compensate for execution time to maintain accurate intervals (bool)

### cleanup_trigger

Clean up the interval trigger.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### setup_trigger

Initialize the interval trigger.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### wait_for_event

Wait for the next interval tick.
**Args:**
- **context (ProcessingContext)**

**Returns:** IntervalTriggerOutput | None


## ManualTrigger

Trigger node that waits for manual events pushed via the API.
This trigger enables interactive workflows where events are pushed
programmatically through the workflow runner's input API. Each event
pushed to the trigger is emitted and processed by the workflow.

This trigger is useful for:
- Building chatbot-style workflows
- Interactive processing pipelines
- Manual batch processing
- Testing and debugging workflows

**Tags:** 

**Fields:**
- **max_events**: Maximum number of events to process (0 = unlimited) (int)
- **name**: Name for this trigger (used in API calls) (str)
- **timeout_seconds**: Timeout waiting for events (None = wait forever) (float | None)

### cleanup_trigger

Clean up the manual trigger.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### push_data

Convenience method to push data as an event.

This wraps the data in the trigger output structure.


**Args:**

- **data**: The data to include in the event.
- **event_type**: The type of event (default: "manual").
**Args:**
- **data (Any)**
- **event_type (str) (default: manual)**

**Returns:** None

### setup_trigger

Initialize the manual trigger.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### wait_for_event

Wait for a manually pushed event.
**Args:**
- **context (ProcessingContext)**

**Returns:** ManualTriggerOutput | None


## WebhookTrigger

Trigger node that starts an HTTP server to receive webhook requests.
Each incoming HTTP request is emitted as an event containing:
- The request body (parsed as JSON if applicable)
- Request headers
- Query parameters
- HTTP method

This trigger is useful for:
- Receiving notifications from external services
- Building API endpoints that trigger workflows
- Integration with third-party webhook providers

**Tags:** 

**Fields:**
- **max_events**: Maximum number of events to process (0 = unlimited) (int)
- **port**: Port to listen on for webhook requests (int)
- **path**: URL path to listen on (str)
- **host**: Host address to bind to. Use '0.0.0.0' to listen on all interfaces. (str)
- **methods**: HTTP methods to accept (list[str])
- **secret**: Optional secret for validating requests (checks X-Webhook-Secret header) (str)

### cleanup_trigger

Stop the HTTP server.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### setup_trigger

Start the HTTP server for receiving webhooks.
**Args:**
- **context (ProcessingContext)**

**Returns:** None

### wait_for_event

Wait for the next webhook request.
**Args:**
- **context (ProcessingContext)**

**Returns:** WebhookTriggerOutput | None


## CopyWorkspaceFile

Copy a file within the workspace.

Use cases:
- Create file backups in workspace
- Duplicate files for different processing
- Copy files to subdirectories

**Tags:** workspace, file, copy, duplicate

**Fields:**
- **source**: Relative source path within workspace (str)
- **destination**: Relative destination path within workspace (str)


## CreateWorkspaceDirectory

Create a directory in the workspace.

Use cases:
- Organize workspace files into directories
- Create output directories for generated files
- Set up workspace structure

**Tags:** workspace, directory, create, folder

**Fields:**
- **path**: Relative path to directory within workspace (str)


## DeleteWorkspaceFile

Delete a file or directory from the workspace.

Use cases:
- Clean up temporary files
- Remove processed files
- Clear workspace data

**Tags:** workspace, file, delete, remove

**Fields:**
- **path**: Relative path to file or directory within workspace (str)
- **recursive**: Delete directories recursively (bool)


## GetWorkspaceDir

Get the current workspace directory path.

Use cases:
- Get the workspace path for reference
- Display workspace location
- Pass workspace path to other nodes

**Tags:** workspace, directory, path

**Fields:**


## GetWorkspaceFileInfo

Get information about a file in the workspace.

Use cases:
- Get file size and timestamps
- Check file type (file vs directory)
- Inspect file metadata

**Tags:** workspace, file, info, metadata

**Fields:**
- **path**: Relative path to file within workspace (str)


## GetWorkspaceFileSize

Get file size in bytes for a workspace file.

Use cases:
- Check file size before processing
- Monitor generated file sizes
- Validate file completeness

**Tags:** workspace, file, size, bytes

**Fields:**
- **path**: Relative path to file within workspace (str)


## IsWorkspaceDirectory

Check if a path in the workspace is a directory.

Use cases:
- Distinguish directories from files
- Validate directory paths
- Filter paths by type

**Tags:** workspace, directory, check, type

**Fields:**
- **path**: Relative path within workspace to check (str)


## IsWorkspaceFile

Check if a path in the workspace is a file.

Use cases:
- Distinguish files from directories
- Validate file types
- Filter paths by type

**Tags:** workspace, file, check, type

**Fields:**
- **path**: Relative path within workspace to check (str)


## JoinWorkspacePaths

Join path components relative to workspace.

Use cases:
- Build file paths within workspace
- Construct nested paths
- Create organized file structures

**Tags:** workspace, path, join, combine

**Fields:**
- **paths**: Path components to join (relative to workspace) (list[str])


## ListWorkspaceFiles

List files in the workspace directory matching a pattern.

Use cases:
- Get files for batch processing within workspace
- Filter workspace files by extension or pattern
- Discover generated files in workspace

**Tags:** workspace, files, list, directory

**Fields:**
- **path**: Relative path within workspace (use . for workspace root) (str)
- **pattern**: File pattern to match (e.g. *.txt, *.json) (str)
- **recursive**: Search subdirectories recursively (bool)

### gen_process

**Args:**
- **context (ProcessingContext)**

**Returns:** typing.AsyncGenerator[nodetool.nodes.nodetool.workspace.ListWorkspaceFiles.OutputType, NoneType]


## MoveWorkspaceFile

Move or rename a file within the workspace.

Use cases:
- Rename files in workspace
- Move files to subdirectories
- Reorganize workspace files

**Tags:** workspace, file, move, rename

**Fields:**
- **source**: Relative source path within workspace (str)
- **destination**: Relative destination path within workspace (str)


## ReadBinaryFile

Read a binary file from the workspace as base64-encoded string.

Use cases:
- Read generated binary data
- Load binary files for processing
- Access non-text files

**Tags:** workspace, file, read, binary

**Fields:**
- **path**: Relative path to file within workspace (str)


## ReadTextFile

Read a text file from the workspace.

Use cases:
- Read configuration files
- Load text data generated by previous nodes
- Process text files in workspace

**Tags:** workspace, file, read, text

**Fields:**
- **path**: Relative path to file within workspace (str)
- **encoding**: Text encoding (utf-8, ascii, etc.) (str)


## SaveImageFile

Save an image to a file in the workspace.

Use cases:
- Save processed images to workspace
- Export edited photos
- Archive image results

**Tags:** workspace, image, save, file, output

**Fields:**
- **image**: The image to save (ImageRef)
- **folder**: Relative folder path within workspace (use . for workspace root) (str)
- **filename**: 
        The name of the image file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)
- **overwrite**: Overwrite the file if it already exists, otherwise file will be renamed (bool)


## SaveVideoFile

Save a video file to the workspace.

Use cases:
- Save processed videos to workspace
- Export video results
- Archive video content

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** workspace, video, save, file, output

**Fields:**
- **video**: The video to save (VideoRef)
- **folder**: Relative folder path within workspace (use . for workspace root) (str)
- **filename**: 
        Name of the file to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)
- **overwrite**: Overwrite the file if it already exists, otherwise file will be renamed (bool)


## WorkspaceFileExists

Check if a file or directory exists in the workspace.

Use cases:
- Validate file presence before processing
- Implement conditional logic based on file existence
- Check for generated files

**Tags:** workspace, file, exists, check

**Fields:**
- **path**: Relative path within workspace to check (str)


## WriteBinaryFile

Write binary data (base64-encoded) to a file in the workspace.

Use cases:
- Save binary data to workspace
- Write decoded base64 data
- Export binary results

**Tags:** workspace, file, write, binary, save

**Fields:**
- **path**: Relative path to file within workspace (str)
- **content**: Base64-encoded binary content to write (str)


## WriteTextFile

Write text to a file in the workspace.

Use cases:
- Save generated text to workspace
- Create configuration files
- Export processed text data

**Tags:** workspace, file, write, text, save

**Fields:**
- **path**: Relative path to file within workspace (str)
- **content**: Text content to write (str)
- **encoding**: Text encoding (utf-8, ascii, etc.) (str)
- **append**: Append to file instead of overwriting (bool)


## Audio

Represents an audio file constant in the workflow.

Use cases:
- Provide a fixed audio input for audio processing nodes
- Reference a specific audio file in the workflow
- Set default audio for testing or demonstration purposes

**Tags:** audio, file, mp3, wav

**Fields:**
- **value** (AudioRef)


## Bool

Represents a boolean constant in the workflow.

Use cases:
- Control flow decisions in conditional nodes
- Toggle features or behaviors in the workflow
- Set default boolean values for configuration

**Tags:** boolean, logic, flag

**Fields:**
- **value** (bool)


## Constant

Base class for fixed-value nodes.
constant, parameter, default

Use cases:
- Provide static inputs to a workflow
- Hold configuration values
- Simplify testing with deterministic outputs

**Tags:** 

**Fields:**


## DataFrame

Represents a fixed DataFrame constant in the workflow.

Use cases:
- Provide static data for analysis or processing
- Define lookup tables or reference data
- Set sample data for testing or demonstration

**Tags:** table, data, dataframe, pandas

**Fields:**
- **value** (DataframeRef)


## Date

Make a date object from year, month, day.

**Tags:** date, make, create

**Fields:**
- **year**: Year of the date (int)
- **month**: Month of the date (int)
- **day**: Day of the date (int)


## DateTime

Make a datetime object from year, month, day, hour, minute, second.

**Tags:** datetime, make, create

**Fields:**
- **year**: Year of the datetime (int)
- **month**: Month of the datetime (int)
- **day**: Day of the datetime (int)
- **hour**: Hour of the datetime (int)
- **minute**: Minute of the datetime (int)
- **second**: Second of the datetime (int)
- **microsecond**: Microsecond of the datetime (int)
- **tzinfo**: Timezone of the datetime (str)
- **utc_offset**: UTC offset of the datetime (int)


## Dict

Represents a dictionary constant in the workflow.

Use cases:
- Store configuration settings
- Provide structured data inputs
- Define parameter sets for other nodes

**Tags:** dictionary, key-value, mapping

**Fields:**
- **value** (dict[str, typing.Any])


## Document

Represents a document constant in the workflow.

**Tags:** document, pdf, word, docx

**Fields:**
- **value** (DocumentRef)


## Float

Represents a floating-point number constant in the workflow.

Use cases:
- Set numerical parameters for calculations
- Define thresholds or limits
- Provide fixed numerical inputs for processing

**Tags:** number, decimal, float

**Fields:**
- **value** (float)


## Image

Represents an image file constant in the workflow.

Use cases:
- Provide a fixed image input for image processing nodes
- Reference a specific image file in the workflow
- Set default image for testing or demonstration purposes

**Tags:** picture, photo, image

**Fields:**
- **value** (ImageRef)


## Integer

Represents an integer constant in the workflow.

Use cases:
- Set numerical parameters for calculations
- Define counts, indices, or sizes
- Provide fixed numerical inputs for processing

**Tags:** number, integer, whole

**Fields:**
- **value** (int)


## JSON

Represents a JSON constant in the workflow.

**Tags:** json, object, dictionary

**Fields:**
- **value** (JSONRef)


## List

Represents a list constant in the workflow.

Use cases:
- Store multiple values of the same type
- Provide ordered data inputs
- Define sequences for iteration in other nodes

**Tags:** array, sequence, collection

**Fields:**
- **value** (list[typing.Any])


## String

Represents a string constant in the workflow.

Use cases:
- Provide fixed text inputs for processing
- Define labels, identifiers, or names
- Set default text values for configuration

**Tags:** text, string, characters

**Fields:**
- **value** (str)


## Video

Represents a video file constant in the workflow.

Use cases:
- Provide a fixed video input for video processing nodes
- Reference a specific video file in the workflow
- Set default video for testing or demonstration purposes

**Tags:** video, movie, mp4, file

**Fields:**
- **value** (VideoRef)


## All

Checks if all boolean values in a list are True.


Use cases:
- Ensure all conditions in a set are met
- Implement comprehensive checks
- Validate multiple criteria simultaneously

**Tags:** boolean, all, check, logic, condition, flow-control, branch

**Fields:**
- **values**: List of boolean values to check (list[bool])


## Compare

Compares two values using a specified comparison operator.

Use cases:
- Implement decision points in workflows
- Filter data based on specific criteria
- Create dynamic thresholds or limits

**Tags:** compare, condition, logic

**Fields:**
- **a**: First value to compare (int | float)
- **b**: Second value to compare (int | float)
- **comparison**: Comparison operator to use (Comparison)


## ConditionalSwitch

Performs a conditional check on a boolean input and returns a value based on the result.

Use cases:
- Implement conditional logic in workflows
- Create dynamic branches in workflows
- Implement decision points in workflows

**Tags:** if, condition, flow-control, branch, true, false, switch, toggle

**Fields:**
- **condition**: The condition to check (bool)
- **if_true**: The value to return if the condition is true (Any)
- **if_false**: The value to return if the condition is false (Any)


## IsIn

Checks if a value is present in a list of options.

Use cases:
- Validate input against a set of allowed values
- Implement category or group checks
- Filter data based on inclusion criteria

**Tags:** membership, contains, check

**Fields:**
- **value**: The value to check for membership (Any)
- **options**: The list of options to check against (list[typing.Any])


## IsNone

Checks if a value is None.

Use cases:
- Validate input presence
- Handle optional parameters
- Implement null checks in data processing

**Tags:** null, none, check

**Fields:**
- **value**: The value to check for None (Any)


## LogicalOperator

Performs logical operations on two boolean inputs.

Use cases:
- Combine multiple conditions in decision-making
- Implement complex logical rules in workflows
- Create advanced filters or triggers

**Tags:** boolean, logic, operator, condition, flow-control, branch, else, true, false, switch, toggle

**Fields:**
- **a**: First boolean input (bool)
- **b**: Second boolean input (bool)
- **operation**: Logical operation to perform (BooleanOperation)


## Not

Performs logical NOT operation on a boolean input.

Use cases:
- Invert a condition's result
- Implement toggle functionality
- Create opposite logic branches

**Tags:** boolean, logic, not, invert, !, negation, condition, else, true, false, switch, toggle, flow-control, branch

**Fields:**
- **value**: Boolean input to negate (bool)


## Some

Checks if any boolean value in a list is True.

Use cases:
- Check if at least one condition in a set is met
- Implement optional criteria checks
- Create flexible validation rules

**Tags:** boolean, any, check, logic, condition, flow-control, branch

**Fields:**
- **values**: List of boolean values to check (list[bool])
