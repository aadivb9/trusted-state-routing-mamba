"""Build figures, tables, HTML, and a review PDF from recorded evidence."""
import csv
import hashlib
import importlib.metadata
from html import escape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'paper'
FIG=OUT/'figures'
FIG.mkdir(exist_ok=True)


def load(name):
    return json.loads((ROOT/name).read_text())


def save_figure(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,2.45));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    box=dict(boxstyle='round,pad=.5',facecolor='#F4F6F8',edgecolor='#444444')
    ax.text(1.6,3,'Committed state\ns',ha='center',va='center',bbox=box)
    ax.text(7.7,3,'Candidate state\nc = F(copy(s), x)',ha='center',va='center',bbox=box)
    ax.annotate('copy + incoming span x',xy=(5.9,3),xytext=(3.2,3),ha='left',va='center',fontsize=9,
                arrowprops=dict(arrowstyle='->',color='#444444'))
    ax.text(5,.65,'Selected state for continuation\ns if rejected; c if accepted',ha='center',va='center',bbox=box)
    ax.annotate('',xy=(3.8,1.25),xytext=(1.6,2.35),arrowprops=dict(arrowstyle='->',color='#4477AA',lw=1.5))
    ax.annotate('',xy=(6.25,1.25),xytext=(7.7,2.35),arrowprops=dict(arrowstyle='->',color='#CC6677',lw=1.5))
    ax.text(1.95,1.55,'reject',color='#4477AA',fontsize=9)
    ax.text(7.1,1.55,'accept',color='#AA4455',fontsize=9)
    ax.text(5,3.85,'One frozen backbone; the external router reads text',ha='center',fontsize=10)
    fig.tight_layout();save_figure(fig,'state_controller')
    a=load('adaptive_router_experiment/results.json')['summary']
    t=load('text_router_benchmark/results.json')['summary']
    g=load('generalization_experiment/results.json')['summary']
    groups=[('Text-router\nindependent (n=100)',t,'malicious'),('Context-aware\nindependent (n=100)',a,'malicious'),('Abstract transfer\nlate attack (n=32)',g,'attack_late')]
    original=[100*s['original'][c]['candidate']['accuracy'] for _,s,c in groups]
    dual=[100*s['dual_state'][c]['candidate']['accuracy'] for _,s,c in groups]
    clean=[100*s['original']['clean']['candidate']['accuracy'] for _,s,_ in groups]
    x=np.arange(len(groups));fig,ax=plt.subplots(figsize=(7,3.6))
    ax.bar(x-.17,original,.32,label='Original under attack',color='#4477AA')
    ax.bar(x+.17,dual,.32,label='Controller under attack',color='#CC6677')
    ax.scatter(x,clean,color='#222222',marker='_',s=350,label='Clean, both',zorder=4)
    for xx,value in zip(x-.17,original):ax.text(xx,value+1.8,f'{value:.1f}%',ha='center',fontsize=9)
    for xx,value in zip(x+.17,dual):ax.text(xx,value+1.8,f'{value:.1f}%',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=[r[0] for r in groups],ylim=(0,110),ylabel='Two-candidate accuracy (%)')
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.22),ncol=3,fontsize=8)
    fig.tight_layout();save_figure(fig,'transfer_results')
    with (OUT/'figure_data.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['evaluation','clean_percent','original_attacked_percent','controller_attacked_percent'])
        writer.writerows((groups[i][0].replace('\n',' '),clean[i],original[i],dual[i]) for i in range(3))
    hardware=load('mamba_28b_feasibility/fp32_run_02/results.json')
    fig,ax=plt.subplots(figsize=(6.5,3.2))
    for mode,label,color in [('original','Original','#4477AA'),('forced_accept','Forced accept','#CC6677'),('filtered','Filtered','#228833')]:
        y=[max(r['peak_allocated_bytes'] for r in hardware['measurements'] if r['mode']==mode and r['prefix_tokens']==n)/2**30 for n in [128,512,2048]]
        ax.plot([128,512,2048],y,'o-',label=label,color=color)
    ax.axhline(hardware['gpu_total_bytes']/2**30,color='black',linestyle='--',label='Device capacity')
    ax.set(xlabel='Prefix tokens',ylabel='Memory (GiB)',xticks=[128,512,2048],ylim=(9.8,15))
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();save_figure(fig,'t4_memory')


def finalize_table():
    path=ROOT/'research_closeout/run_03/results.json'
    if not path.exists():return
    data=json.loads(path.read_text())
    assert data['status']=='complete'
    audit=load('research_closeout/run_03/verification.json')
    assert audit['status']=='passed'
    lines=['The post-development closing diagnostic completed all 600 reserved baseline examples. It retains the original failed competence gate and does not include attack-conditioned generations.','',
           '| Role | Task | Published parser correct | Strict parser correct | Strict valid |',
           '| --- | --- | ---: | ---: | ---: |']
    for role in ['target','injected']:
        for task in ['sst2','sms_spam','mrpc']:
            s=data['summary'][role+'/'+task]
            lines.append(f"| {role.capitalize()} | {task} | {s['official_correct']}/100 | {s['strict_correct']}/100 | {s['strict_valid']}/100 |")
    lines+=['','The target role measures the legitimate task on its reserved inputs. The injected role measures the attacker task on its own clean inputs, not attack success. The roles differ in source records and published instruction variants, so their difference does not isolate a causal prompt effect. Each task/role sample is balanced and selected by the published deterministic procedure. These are descriptive results for the selected records, not random population samples. Formatting and task dependence must be considered alongside accuracy.']
    text=(OUT/'manuscript.md').read_text()
    text=re.sub(r'<!-- FINAL_BASELINE_TABLE_START -->.*?<!-- FINAL_BASELINE_TABLE_END -->',
                '<!-- FINAL_BASELINE_TABLE_START -->\n'+'\n'.join(lines)+'\n<!-- FINAL_BASELINE_TABLE_END -->',text,flags=re.S)
    (OUT/'manuscript.md').write_text(text)
    (OUT/'final_baseline_table.md').write_text('\n'.join(lines)+'\n')


def inline(node):
    text=escape(node.text or '')
    for child in node:
        tag={'strong':'b','em':'i','code':'font'}.get(child.tag,child.tag)
        attrs=' name="Mono"' if child.tag=='code' else ''
        if child.tag=='a':attrs=' href="'+escape(child.get('href',''),quote=True)+'" color="#225588"'
        text+=f'<{tag}{attrs}>'+inline(child)+f'</{tag}>'+escape(child.tail or '')
    return text


def render():
    text=(OUT/'manuscript.md').read_text()
    text=text.replace('### 3.2. Routers',
                      '![State controller](figures/state_controller.png)\n\n*Figure 1. Both cache banks share frozen model weights. An external text router selects the preserved or candidate state; the evaluated router does not inspect hidden-state anomalies.*\n\n### 3.2. Routers')
    text=text.replace('### 5.2. Out-of-schema failure and filtering controls',
                      '![Transfer results](figures/transfer_results.png)\n\n*Figure 2. Separate evaluation settings; all scores are two-candidate accuracy. Clean reference marks are shared by both variants. The settings differ in data and router; their percentages are not pooled.*\n\n### 5.2. Out-of-schema failure and filtering controls')
    html=markdown.markdown(text,extensions=['tables'])
    (OUT/'manuscript.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Trusted-State Routing</title><style>body{max-width:850px;margin:48px auto;font:17px/1.6 Georgia,serif;padding:0 24px}h1,h2,h3{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ccc;padding:8px;text-align:left}img{max-width:100%}a{color:#225588}</style></head><body>'+html+'</body></html>')
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('Body','DejaVuSerif.ttf'),('Body-Bold','DejaVuSerif-Bold.ttf'),('Body-Italic','DejaVuSerif-Italic.ttf'),('Mono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic',boldItalic='Body-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PaperBody',fontName='Body',fontSize=9.7,leading=14,spaceAfter=7))
    styles.add(ParagraphStyle(name='PaperTitle',fontName='Body-Bold',fontSize=17,leading=22,spaceAfter=14))
    styles.add(ParagraphStyle(name='PaperH2',fontName='Body-Bold',fontSize=12,leading=16,spaceBefore=12,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='PaperH3',fontName='Body-Bold',fontSize=10.5,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True))
    styles.add(ParagraphStyle(name='PaperCell',fontName='Body',fontSize=7.2,leading=10))
    story=[]
    root=ET.fromstring('<root>'+html+'</root>')
    for element in root:
        if element.tag in ['h1','h2','h3','p']:
            img=element.find('img')
            if img is not None:
                image=Image(str(OUT/img.get('src')));image.drawHeight*=6.7*inch/image.drawWidth;image.drawWidth=6.7*inch
                story.extend([Spacer(1,8),image,Spacer(1,8)]);continue
            style={'h1':'PaperTitle','h2':'PaperH2','h3':'PaperH3','p':'PaperBody'}[element.tag]
            story.append(Paragraph(inline(element),styles[style]))
        elif element.tag=='table':
            rows=[[Paragraph(inline(cell),styles['PaperCell']) for cell in row] for row in element.findall('.//tr')]
            n=len(rows[0]);widths=[6.8*inch/n]*n
            if n==5:widths=[2.1*inch,1.0*inch,1.1*inch,1.1*inch,1.5*inch]
            if n==5 and ''.join(element.find('.//tr')[0].itertext())=='Role':
                widths=[.8*inch,1.3*inch,1.7*inch,1.6*inch,1.4*inch]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EDF1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,1),(-1,-1),.25,colors.lightgrey),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            block=[table,Spacer(1,9)]
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name in ['PaperH2','PaperH3']:
                block.insert(0,story.pop())
            story.append(KeepTogether(block))
    def footer(canvas,doc):
        canvas.setFont('Body',8);canvas.drawString(.85*inch,.45*inch,'Research draft — not submission formatted');canvas.drawRightString(7.65*inch,.45*inch,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'manuscript.pdf'),pagesize=(8.5*inch,11*inch),rightMargin=.85*inch,leftMargin=.85*inch,topMargin=.75*inch,bottomMargin=.75*inch,title='Trusted-State Routing in Frozen Mamba',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Built manuscript.pdf, manuscript.html, figures and figure_data.csv')


def record_build():
    inputs = ['paper/manuscript.md', 'paper/build.py', 'paper/references.bib',
              'paper/build_requirements.txt', 'adaptive_router_experiment/results.json',
              'text_router_benchmark/results.json', 'generalization_experiment/results.json',
              'mamba_28b_feasibility/fp32_run_02/results.json',
              'research_closeout/run_03/results.json', 'research_closeout/run_03/verification.json']
    outputs = ['paper/manuscript.pdf', 'paper/manuscript.html', 'paper/figure_data.csv',
               'paper/final_baseline_table.md']
    outputs += [str(p.relative_to(ROOT)) for p in sorted(FIG.iterdir()) if p.suffix in ('.pdf', '.png')]
    digest = lambda names: {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    record = dict(inputs=digest(inputs), outputs=digest(outputs),
                  environment={p: importlib.metadata.version(p) for p in ('markdown', 'matplotlib', 'numpy', 'reportlab')},
                  scope='Build provenance and byte freshness; does not replace visual or semantic review.')
    (OUT/'build_verification.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    finalize_table();figures();render();record_build()
