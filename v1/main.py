import argparse

import tarot


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='Tarot',
                    description='Trains and detects Tarot cards from images',
                    epilog='Your future awaits!!')
    
    parser.add_argument('dataset')              # positional argument
    parser.add_argument('action')              # positional argument
    parser.add_argument('-d', '--directory')    # option that takes a value
    parser.add_argument('-f', '--filename')    # option that takes a value
    parser.add_argument('-a', '--append', action='store_true')  # on/off flag
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag

    args = parser.parse_args()

    dataset_file = f'{args.dataset.replace(' ', '_')}.json'

    if "train" == args.action.lower():
        model = tarot.ImageModel()
        if args.append:
            model = tarot.load(dataset_file)
        model.train(args.directory)
        model.export(dataset_file)

    elif "identify" == args.action.lower():
        model = tarot.load(dataset_file)
        card_name, similarity_score = model.identify(args.filename)
        print(f"Identified Card: {card_name}")
        print(f"Similarity Score: {similarity_score}")

    elif "detect" == args.action.lower():
        model = tarot.load(dataset_file)
        for card_name, similarity_score in model.detect(args.filename):
            print(f"Identified Card: {card_name}")
            print(f"Similarity Score: {similarity_score}")



'''

python main.py --directory ../data/dataset tarot train


python main.py --filename ../data/dataset/2_of_cups/2_of_cups.jpg tarot identify


python main.py --filename ../data/dataset/2_of_cups/2_of_cups.jpg tarot detect

python main.py --filename ../data/tests/20241130_135912.jpg tarot detect




'''