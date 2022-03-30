from argparse import ArgumentParser
import json
from tqdm import tqdm

def create_dataset(question_file, answer_file, outpath):
    questions_data = json.load(open(question_file))['questions']
    answers_data = json.load(open(answer_file))['annotations']
    with open(outpath, 'w') as writer:
        for q_data, a_data in tqdm(zip(questions_data, answers_data)):
            assert q_data['question_id'] == a_data['question_id']
            answer = a_data['multiple_choice_answer']
            image_id = str(a_data['image_id'])
            image_id = '0' * (12 - len(image_id)) + image_id + '.jpg'
            question_id = a_data['question_id']
            question = q_data['question']
            writer.write(f'q&a.{question_id}\t{question}\t{question}\t0-{len(question)}\t{image_id}\t{answer}\t{image_id}\n')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--questions_file', required=True)
    parser.add_argument('--answers_file', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()
    create_dataset(args.questions_file, args.answers_file, args.output_file)