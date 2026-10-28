"""Freeze a V2 blinded packet before independent review; never send it."""
import hashlib,json,secrets,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.validation.blinded_review_v2 import make_packet


def main():
    if (OUT/'reviewer_packet.json').exists():raise RuntimeError('Preserve the existing frozen reviewer packet')
    source=ROOT/'archive/sealed_workspace/generalization_experiment/dataset.json'
    records=json.loads(source.read_text())['records']
    packet,private=make_packet(records,secrets.randbits(256))
    (OUT/'private').mkdir(exist_ok=True)
    (OUT/'reviewer_packet.json').write_text(json.dumps(packet,indent=2)+'\n')
    (OUT/'private/mapping.json').write_text(json.dumps(private,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'reviewer_packet.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(OUT/'reviewer_packet.json','reviewer_packet.json')
    status={'status':'awaiting_independent_review','packet_version':2,'items':len(records),'notes':sum(len(x['notes']) for x in packet['items']),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'packet_sha256':hashlib.sha256((OUT/'reviewer_packet.json').read_bytes()).hexdigest(),
            'private_mapping_sha256':hashlib.sha256((OUT/'private/mapping.json').read_bytes()).hexdigest(),
            'seed_policy':'256-bit random seed retained only in private mapping; IDs generated independently with secrets.token_hex.',
            'reviewer_input_excludes_seed_and_mapping':True,'independent_review_completed':False,'human_spotcheck_completed':False,
            'scope':'V1 public-seed export withdrawn before review; V2 packet and mapping are frozen. Export is not independent annotation validation.'}
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))


if __name__=='__main__':main()
