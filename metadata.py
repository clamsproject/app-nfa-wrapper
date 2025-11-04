"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
import pathlib

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from lapps.discriminators import Uri


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification.
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """

    nemo_version_file = pathlib.Path(__file__).parent / 'NEMO_VERSION'
    try:
        analyzer_version = nemo_version_file.read_text().strip()
    except FileNotFoundError:
        analyzer_version = 'main'

    metadata = AppMetadata(
        name="CLAMS NFA Wrapper",
        description="Wraps the [NVIDIA NeMo Forced Aligner tool](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)"
                    "to temporally align transcribed text with its audio source. "
                    "Requires a local [NeMo](https://github.com/NVIDIA/NeMo#installation) installation.",
        app_license="Apache 2.0",
        identifier="nfa-wrapper",
        url="https://github.com/clamsproject/app-nfa-wrapper",
        analyzer_version=analyzer_version,
        analyzer_license="Apache 2.0",
    )

    # Input Spec
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)
    in_txt = metadata.add_input(DocumentTypes.TextDocument)
    in_txt.add_description('Text content transcribed from audio input with no existing annotations.')

    # Output Spec
    out_tkn = metadata.add_output(Uri.TOKEN)
    out_tkn.add_description('Token from original text split on whitespace. `word` property stores the string value '
                            'of the token. `start` and `end` properties indicate position of token in entire text. '
                            '`document` property identifies source text document.')
    out_tf = metadata.add_output(AnnotationTypes.TimeFrame, frameType='speech', timeUnit='milliseconds')
    out_tf.add_description('TimeFrame annotation representing the source audio segment corresponding to a given '
                           'transcribed token, with `start` and `end` times given in milliseconds.')
    out_ali = metadata.add_output(AnnotationTypes.Alignment)
    out_ali.add_description('Alignment between `Token` and `TimeFrame` annotations.')

    # Parameters
    metadata.add_parameter(name='model', description='NeMo ASR model to use. Choices: fc_hybrid, '
                                                     'parakeet, conformer, fc_ctc. '
                                                     'By default, the fc_hybrid model will be used.',
                           type='string',
                           choices=["fc_hybrid",
                                    "parakeet", "conformer", "fc_ctc"],
                           default="fc_hybrid")

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
