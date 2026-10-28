"""Produce a local blinded packet and separately retained mapping; do not send it."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.validation.blinded_review import make_packet


def main():
    if (OUT/'reviewer_packet.json').exists():raise RuntimeError('Preserve the frozen packet; do not overwrite')
    source=ROOT/'archive/sealed_workspace/generalization_experiment/dataset.json'
    records=json.loads(source.read_text())['records']
    packet,private=make_packet(records,20261022)
    private_path=OUT/'private';private_path.mkdir(exist_ok=True)
    (OUT/'reviewer_packet.json').write_text(json.dumps(packet,indent=2)+'\n')
    (private_path/'mapping.json').write_text(json.dumps(private,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'reviewer_packet.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(OUT/'reviewer_packet.json','reviewer_packet.json')
    status={'status':'awaiting_independent_review','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'packet_sha256':hashlib.sha256((OUT/'reviewer_packet.json').read_bytes()).hexdigest(),
            'private_mapping_sha256':hashlib.sha256((private_path/'mapping.json').read_bytes()).hexdigest(),
            'items':len(records),'notes':sum(len(x['notes']) for x in packet['items']),
            'reviewer_input_excludes_seed_and_mapping':True,'independent_review_completed':False,'human_spotcheck_completed':False,
            'scope':'Blinding machinery and packet preparation only. No annotation validation is claimed until an independent review is returned.'}
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))


if __name__=='__main__':main()
