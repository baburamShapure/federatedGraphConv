import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--num_sample', 
                    default= 128, 
                    help =  'Number of samples in each window')

parser.add_argument('--dist_thresh', 
                    default= 0.3, 
                    help =  'Minimum euclidean distance to draw an edge')

parser.add_argument('--train_prop', 
                    default= 0.7, 
                    help =  'Proportion of data to include in training.')

parser.add_argument('--epochs', 
                    default= 1000, 
                    help = 'Number of epochs to run')

parser.add_argument('--batch_size', 
                    default= 4, 
                    help = 'Batch size in each iteration')

parser.add_argument('--lr', 
                    default= 0.01, 
                    help = 'Learning rate')


if __name__ == '__main__': 
    args = parser.parse_args()
    print(args.num_sample)
    print(args.lr)