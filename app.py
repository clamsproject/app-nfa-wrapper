"""
CLAMS wrapper for NVIDIA NeMo Forced Aligner text-to-audio timestamp alignment. Requires local NeMo installation.
"""

import argparse
import json
import logging
import tempfile
import ffmpeg
import subprocess
import os

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes
from clams import ClamsApp, Restifier
from mmif import Mmif, AnnotationTypes, DocumentTypes
# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri

# imports for NeMo
from align import main, AlignmentConfig, ASSFileConfig  # maybe don't need ASSFileConfig

# global dict for model options - pending further changes
MODEL_OPTIONS = {
    "fc_hybrid": "stt_en_fastconformer_hybrid_large_pc",
    "parakeet": "parakeet-tdt_ctc-110m",
    "conformer": "stt_en_conformer_ctc_medium",
    "fc_ctc": "stt_en_fastconformer_ctc_large"
}

class NfaWrapper(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # using metadata.py
        pass

    @staticmethod
    def convert_to_16k_wav_bytes(input_path):
        """
        Converts an audio or video file to 16kHz mono WAV format using ffmpeg-python.
        Returns the WAV data as bytes (in-memory, no output file).
        """
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_wav.close()  # Close the file so ffmpeg can write to it
        ffmpeg.input(input_path).output(temp_wav.name, format='wav', ac=1, ar=16000).overwrite_output().run()
        return temp_wav.name  # Return the path to the temporary WAV file

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        mmif = mmif if isinstance(mmif, Mmif) else Mmif(mmif)

        view = mmif.new_view()
        self.sign_view(view, parameters)
        view.new_contain(AnnotationTypes.TimeFrame, frameType='speech', timeUnit='milliseconds')
        view.new_contain(AnnotationTypes.Alignment, sourceType=Uri.TOKEN, targetType=AnnotationTypes.TimeFrame)

        # TODO (ldial @ 8/2/25): add error handling for empty inputs?
        # get audio source from input MMIF and convert to 16kHz mono WAV format
        audio_sources = mmif.get_documents_by_type(DocumentTypes.AudioDocument) + mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        audio = audio_sources[0]
        audio_path = audio.location_path()
        resampled_path = self.convert_to_16k_wav_bytes(audio_path)

        transcript = mmif.get_documents_by_type(DocumentTypes.TextDocument)[0]
        transcript_text = transcript.text_value

        # assemble NeMo-style manifest and create json file
        input_data = {"audio_filepath": resampled_path, "text": transcript_text}
        manifest = tempfile.NamedTemporaryFile(mode='w+t', delete=False, suffix='.json')
        json.dump(input_data, manifest)
        manifest.close()
        # create a temporary directory to pass to align.py for output files
        tmpdir = tempfile.TemporaryDirectory()

        # get model name based on user input
        model_name = parameters.get("model")
        if model_name not in MODEL_OPTIONS:
            raise ValueError("Unsupported model; note that this wrapper does not "
                             "support all NeMo models. See parameters specification in the appmetadata.")
        else:
            model_name = MODEL_OPTIONS[model_name]
        # get paths of temporary manifest and output directory
        manifest_path = manifest.name
        output_dir = tmpdir.name
        output_format = ['ctm']
        # call align.py with necessary arguments
        alignment_config = AlignmentConfig(
			pretrained_name=model_name,
			manifest_filepath=manifest_path,
			output_dir=output_dir,
			audio_filepath_parts_in_utt_id=1,
			batch_size=1,
			use_local_attention=True,
			additional_segment_grouping_separator="|",
			# transcribe_device='cpu',
			# viterbi_device='cpu',
			save_output_file_formats=output_format,
		)
        main(alignment_config)

        # get file name for word-level CTM output
        resampled_name = resampled_path.split('/')[-1]
        target_file = resampled_name.split('.')[0] + ".ctm"
        char_offset = 0
        # retrieve info from CTM file
        with open(tmpdir.name + "/ctm/words/" + target_file) as results:
            # TODO (ldial @ 8/2/25): catch empty output errors?
            for line in results:
                data = line.split()
                start_time = float(data[2])
                duration = float(data[3])
                word = data[4]

                # find token position in entire text
                tok_start = transcript_text.index(word, char_offset)
                tok_end = tok_start + len(word)
                char_offset = tok_end
                token = view.new_annotation(Uri.TOKEN, word=word, start=tok_start, end=tok_end, document=transcript.id)

                # start and duration are in seconds, so convert to ms for timestamping
                tf_start = int(start_time * 1000.0)
                tf_end = tf_start + int(duration * 1000.0)
                tf = view.new_annotation(AnnotationTypes.TimeFrame, start=tf_start, end=tf_end)
                view.new_annotation(AnnotationTypes.Alignment, source=tf.id, target=token.id)

        # clean up temporary directory just in case, but might not be necessary
        tmpdir.cleanup()

        return mmif

def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    return NfaWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
