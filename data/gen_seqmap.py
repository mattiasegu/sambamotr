import os
import argparse


def write_folder_names_to_file(directory, outfile, include):
    # Get a list of names in the directory
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    # Open the file in write mode
    with open(outfile, 'w') as file:
        # Write the header
        file.write('name\n')
        # Write each folder name to the file
        for folder in folder_names:
            if include in folder:
                file.write(folder + '\n')

# Example usage:
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='List folders in a directory and write them to a file.')
    # Add arguments
    parser.add_argument('--data-dir', type=str, help='Directory to list folders from')
    parser.add_argument('--split', type=str, help='Split to generate seqmap for')
    parser.add_argument('--include', default='', type=str, help='Keyword necessary to include dir')
    parser.add_argument('--exclude', default='', type=str, help='Exclude directories containing this keyword')


    # Parse arguments
    args = parser.parse_args()

    # Use the arguments
    write_folder_names_to_file(directory=os.path.join(args.data_dir, args.split),
                               outfile=os.path.join(args.data_dir, f'{args.split}_seqmap.txt'),
                               include=args.include)

if __name__ == '__main__':
    main()