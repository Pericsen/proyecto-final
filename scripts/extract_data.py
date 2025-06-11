import argparse
from src.extract.facebook_extractor import FacebookExtractor
from src.extract.instagram_extractor import InstagramExtractor
from src.extract.munidigital_extractor import MuniDigitalExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['facebook', 'instagram', 'munidigital', 'all'])
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    
    extractors = {
        'facebook': FacebookExtractor(),
        'instagram': InstagramExtractor(), 
        'munidigital': MuniDigitalExtractor()
    }
    
    if args.source == 'all':
        for extractor in extractors.values():
            extractor.extract_and_save(limit=args.limit)
    else:
        extractors[args.source].extract_and_save(limit=args.limit)

if __name__ == '__main__':
    main()