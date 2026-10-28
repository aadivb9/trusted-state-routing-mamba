"""Import a returned review with reviewer attestation; never fabricate one."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.validation.blinded_review import validate_reviews


def main():
    parser=argparse.ArgumentParser();parser.add_argument('response',type=Path);args=parser.parse_args()
    output=OUT/'returned_review.json'
    if output.exists():raise RuntimeError('Preserve returned reviews; record another version separately')
    source=json.loads(args.response.read_text());reviewer=source['reviewer']
    if not reviewer.get('name') or reviewer.get('independent_of_generation') is not True:
        raise ValueError('Named independent-review attestation required')
    if any(not note.get('reason','').strip() for review in source['reviews'] for note in review['notes']):
        raise ValueError('A source-grounded reason is required for each returned note label')
    packet=json.loads((OUT/'reviewer_packet.json').read_text());private=json.loads((OUT/'private/mapping.json').read_text())
    result=validate_reviews(packet,private,source['reviews'])
    result.update(reviewer=reviewer,response_sha256=hashlib.sha256(args.response.read_bytes()).hexdigest(),
                  status='review_received',human_spotcheck_completed=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
