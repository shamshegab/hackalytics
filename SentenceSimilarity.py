import tensorflow_hub as hub
import numpy as np

def sentence_similarity(answers):
    
    assert len(answers) == 2
    assert "Model_Answer" in answers
    assert "Applicant_Answer" in answers

    module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
    model = hub.load(module_url)

    answers_list = [a for a in answers.values()]
    embeddings = model(answers_list)
    similarity = np.inner(embeddings, embeddings)

    return similarity[0][1]


if __name__ == "__main__":
    applicant_1 = {
        "Model_Answer": "Machine Learning Problems can be Supervised or Unsupervised",
        "Applicant_Answer": "Machine Learning problems can be Linear Regression, Logistic Regression or Neural Networks"
    }

    rate_1 = sentence_similarity(applicant_1)
    print(f"Applicant 1 answer has a similarity rating of {rate_1}")

    applicant_2 = {
        "Model_Answer": "Machine Learning Problems can be Supervised or Unsupervised",
        "Applicant_Answer": "Machine Learning use-cases can be either Supervised use case like Classification or can be a Reinforcement Learning use case"
    }

    rate_2 = sentence_similarity(applicant_2)
    print(f"Applicant 2 answer has a similarity rating of {rate_2}")

    applicant_3 = {
        "Model_Answer": "Machine Learning Problems can be Supervised or Unsupervised",
        "Applicant_Answer": "Machine Learning can be numpy, pandas, sklearn or excel"
    }

    rate_3 = sentence_similarity(applicant_3)
    print(f"Applicant 3 answer has a similarity rating of {rate_3}")

    applicant_4 = {
        "Model_Answer": "Machine Learning Problems can be Supervised or Unsupervised",
        "Applicant_Answer": "I go to school by bus"
    }

    rate_4 = sentence_similarity(applicant_4)
    print(f"Applicant 4 answer has a similarity rating of {rate_4}")