
from transformers import pipeline
from evaluator import Evaluator


class FluencyEvaluator(Evaluator):
    model =  pipeline(
        'audio-classification',
        model='JohnJumon/fluency_accuracy',
        chunk_length_s=30,
        device=1
    )
    score_map = {
        'Very Influent': 1,
        'Influent': 2,
        'Average': 3,
        'Fluent': 4
    }


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from preprocess import get_audio_metadata

    location = "./SpeakNow_test_data.csv"
    speaknow = pd.read_csv(location)
    
    metadata = get_audio_metadata(location)

    scores, confidences = FluencyEvaluator()(
        metadata.file_name.values.tolist(),
        metadata.assessment_id.values.tolist(),
        score_type='argmax',
        confidence_type='entropy'
    )

    speaknow_scores = dict()
    for id in scores:
        scores[id] = np.sum(np.array(scores[id])*np.array(confidences[id])) / np.sum(np.array(confidences[id]))
        confidences[id] = np.max(confidences[id])
        speaknow_scores[id] = speaknow[speaknow['assessment_id'] == id]['fluency_avg'].values[0]

    plt.cla()
    plt.scatter(speaknow_scores.values(), scores.values())
    plt.savefig('./out/fluency_v_sn.png')

    plt.cla()
    plt.scatter(scores.values(), confidences.values())
    plt.savefig('./out/fluency_v_conf.png')
    