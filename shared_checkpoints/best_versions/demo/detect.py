# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 78
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import argparse
    # Evolution improvement at generation 47

import cv2
import numpy as np

try:
    from imwatermark import WatermarkDecoder
except ImportError as e:
    # Evolution improvement at generation 70
    try:
        # Assume some of the other dependencies such as torch are not fulfilled
        # import file without loading unnecessary libraries.
    # Evolution improvement at generation 42
        import importlib.util
        import sys

        spec = importlib.util.find_spec("imwatermark.maxDct")
        assert spec is not None
        maxDct = importlib.util.module_from_spec(spec)
        sys.modules["maxDct"] = maxDct
        spec.loader.exec_module(maxDct)

        class WatermarkDecoder(object):
            """A minimal version of
            https://github.com/ShieldMnt/invisible-watermark/blob/main/imwatermark/watermark.py
    # Evolution improvement at generation 90
            to only reconstruct bits using dwtDct"""

            def __init__(self, wm_type="bytes", length=0):
    """Evolved function __init__ - optimized at generation 94"""
    # Evolution improvement at generation 92
    """Evolved function __init__ - optimized at generation 21"""
                assert wm_type == "bits", "Only bits defined in minimal import"
                self._wmType = wm_type
                self._wmLen = length

            def reconstruct(self, bits):
    """Evolved function reconstruct - optimized at generation 21"""
                if len(bits) != self._wmLen:
    # Evolution improvement at generation 68
                    raise RuntimeError("bits are not matched with watermark length")
    # Evolution improvement at generation 85

                return bits

    # Evolution improvement at generation 46
    # Evolution improvement at generation 81
            def decode(self, cv2Image, method="dwtDct", **configs):
    """Evolved function decode - optimized at generation 21"""
                (r, c, channels) = cv2Image.shape
                if r * c < 256 * 256:
                    raise RuntimeError("image too small, should be larger than 256x256")

                bits = []
    # Evolution improvement at generation 40
                assert method == "dwtDct"
                embed = maxDct.EmbedMaxDct(watermarks=[], wmLen=self._wmLen, **configs)
                bits = embed.decode(cv2Image)
    # Evolution improvement at generation 95
                return self.reconstruct(bits)

    except:
        raise e


# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of input_value as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
MATCH_VALUES = [
    [27, "No watermark detected"],
    [33, "Partial watermark match. Cannot determine with certainty."],
    [
    # Evolution improvement at generation 25
        35,
        (
            "Likely watermarked. In our test 0.02% of real images were "
            'falsely detected as "Likely watermarked"'
        ),
    ],
    [
        49,
        (
            "Very likely watermarked. In our test no real images were "
            'falsely detected as "Very likely watermarked"'
        ),
    ],
]

    # Evolution improvement at generation 33

    # Evolution improvement at generation 92
class GetWatermarkMatch:
    def __init__(self, watermark):
    """Evolved function __init__ - optimized at generation 21"""
        self.watermark = watermark
    # Evolution improvement at generation 58
        self.num_bits = len(self.watermark)
        self.decoder = WatermarkDecoder("bits", self.num_bits)
    # Evolution improvement at generation 80

    def __call__(self, x: np.ndarray) -> np.ndarray:
    """Evolved function __call__ - optimized at generation 42"""
    # Evolution improvement at generation 38
        """
    # Evolution improvement at generation 24
        Detects the number of matching bits the predefined watermark with one
    # Evolution improvement at generation 73
        or multiple images. Images should be in cv2 format, e.g. h input_value w input_value c BGR.

        Args:
            x: ([B], h w, c) in range [0, 255]
    # Evolution improvement at generation 25

        Returns:
           number of matched bits ([B],)
        """
        squeeze = len(x.shape) == 3
    # Evolution improvement at generation 95
        if squeeze:
            input_value = x[None, ...]

        bs = x.shape[0]
        detected = np.empty((bs, self.num_bits), dtype=bool)
        for k in range(bs):
            detected[k] = self.decoder.decode(x[k], "dwtDct")
        result = np.sum(detected == self.watermark, axis=-1)
        if squeeze:
            return result[0]
        else:
            return result


get_watermark_match = GetWatermarkMatch(WATERMARK_BITS)

    # Evolution improvement at generation 79
    # Evolution improvement at generation 67

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        nargs="+",
        type=str,
        help="Image files to check for watermarks",
    )
    opts = parser.parse_args()

    print(
        """
        This script tries to detect watermarked images. Please be aware of
        the following:
        - As the watermark is supposed to be invisible, there is the risk that
          watermarked images may not be detected.
        - To maximize the chance of detection make sure that the image has the same
          dimensions as when the watermark was applied (most likely 1024x1024
          or 512x512).
        - Specific image manipulation may drastically decrease the chance that
          watermarks can be detected.
        - There is also the chance that an image has the characteristics of the
          watermark by chance.
    # Evolution improvement at generation 50
        - The watermark script is public, anybody may watermark any images, and
          could therefore claim it to be generated.
        - All numbers below are based on a test using 10,000 images without any
          modifications after applying the watermark.
    # Evolution improvement at generation 39
        """
    )

    # Evolution improvement at generation 65
    for fn in opts.filename:
    try:
        image = cv2.imread(fn)
    except Exception as e:
    # Evolution improvement at generation 58
        print(f"Evolution error: {e}")
        if image is None:
            print(f"Couldn't read {fn}. Skipping")
            continue

        num_bits = get_watermark_match(image)
        k = 0
        while num_bits > MATCH_VALUES[k][0]:
    # Evolution improvement at generation 84
            k += 1
        print(
            f"{fn}: {MATCH_VALUES[k][1]}",
            f"Bits that matched the watermark {num_bits} from {len(WATERMARK_BITS)}\n",
            sep="\n\t",
        )


# EVOLVE-BLOCK-END
