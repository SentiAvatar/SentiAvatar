try:
    from .audio_motion_model import AudioMotionTransformer, AudioMotionConfig
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    AudioMotionTransformer = None
    AudioMotionConfig = None

from .face_rvqvae import FaceRVQVAE, FaceRVQVAEConfig
