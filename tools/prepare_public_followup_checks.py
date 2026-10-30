"""Add narrow follow-up checks to an existing separate public export."""
import argparse
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    source=Path(__file__).resolve().parents[1];dest=args.destination.resolve()
    if dest==source or not (dest/'provenance/public_manifest.json').is_file():raise ValueError('Expected separate existing public export')
    shutil.copy2(source/'tools/verify_followup_public.py',dest/'tools/verify_followup_public.py')
    p=dest/'tools/verify_public.py';text=p.read_text()
    if 'followup = verify_followup(ROOT)' not in text:
        needle="    print(json.dumps({'status': 'passed', 'exported_files_checked': len(manifest),"
        assert text.count(needle)==1,'Unknown public checker; review before editing'
        text=text.replace(needle,"    sys.path.insert(0,str(ROOT/'tools'))\n    from verify_followup_public import verify as verify_followup\n    followup = verify_followup(ROOT)\n"+needle)
        text=text.replace("'scope': 'Public bytes, syntax, links and included saved-output checks only.","'followup':followup,\n                      'scope': 'Public bytes, syntax, links and included saved-output checks only.")
    if 'Public manifest inventory coverage differs' not in text:
        needle="    for name, expected in manifest.items():"
        insert="""    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
            and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts
            and p.name!='.DS_Store' and p.suffix!='.pyc'
            and str(p.relative_to(ROOT))!='provenance/public_manifest.json'}
    require(actual==set(manifest),'Public manifest inventory coverage differs')
    for name in manifest:
        require(not (name.startswith('experiments/19_embedding_followup/raw/') and name!='experiments/19_embedding_followup/raw/provenance.json'),'Raw new BIPIA corpus in public export')
        require(name not in ['experiments/19_embedding_followup/prepared.json','experiments/19_embedding_followup/derived_official_test.jsonl'],'Raw prepared corpus in public export')
        require(not name.startswith(('experiments/20_sst2_followup/progress_snapshots/','experiments/20_sst2_followup/continuation_progress/')),'Raw response snapshot in public export')
        require(name not in ['experiments/20_sst2_followup/'+n for n in ['prepared.json','masks.json','screen_results.json','final_results.json','continuation_results.json','continuation_overlap.json']],'Raw new source/response material in public export')
"""
        assert text.count(needle)==1;text=text.replace(needle,insert+needle)
    p.write_text(text)
    p=dest/'tools/run_public_checks.py';text=p.read_text()
    if "'test_followup_serialization.py'" not in text:text=text.replace("'test_remediation_timing.py', 'test_public_extractor.py'","'test_remediation_timing.py', 'test_public_extractor.py','test_followup_serialization.py'")
    p.write_text(text)
    p=dest/'.gitignore';text=p.read_text();extra='''\n# New benchmark source text and raw responses stay in the local research package.
/experiments/19_embedding_followup/prepared.json
/experiments/19_embedding_followup/derived_official_test.jsonl
/experiments/19_embedding_followup/raw/*
!/experiments/19_embedding_followup/raw/provenance.json
/experiments/20_sst2_followup/prepared.json
/experiments/20_sst2_followup/masks.json
/experiments/20_sst2_followup/screen_results.json
/experiments/20_sst2_followup/final_results.json
/experiments/20_sst2_followup/continuation_results.json
/experiments/20_sst2_followup/continuation_overlap.json
/experiments/20_sst2_followup/progress_snapshots/
/experiments/20_sst2_followup/continuation_progress/
'''
    if extra not in text:p.write_text(text+extra)


if __name__=='__main__':main()
