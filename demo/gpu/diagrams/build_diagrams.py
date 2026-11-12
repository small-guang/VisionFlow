"""Generate architecture diagrams; these are explanations, never inference evidence."""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H = 1600, 900
BG, PANEL, LINE = '#0b141b', '#142630', '#29424d'
WHITE, MUTED, GREEN, AMBER = '#e9f5f2', '#a1b8c2', '#55dfbb', '#ffd177'


def build(font_path, name, title, subtitle, nodes, footer, color=GREEN, note=None):
    image = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(image)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9Z" fill="'+color+'"/></marker></defs>',
           f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    def text(x, y, value, size=28, fill=WHITE):
        font = ImageFont.truetype(str(font_path), size)
        try: font.set_variation_by_axes([500])
        except OSError: pass
        if draw.textlength(value, font=font) > W-x-50:
            raise ValueError(value)
        draw.text((x,y),value,font=font,fill=fill)
        svg.append(f'<text x="{x}" y="{y+size}" fill="{fill}" font-family="Noto Sans SC,sans-serif" font-size="{size}" font-weight="500">{escape(value)}</text>')
    text(60,35,'VISIONFLOW / GPU VIDEO DEMO',22,MUTED)
    text(60,90,title,48)
    text(60,165,subtitle,26,color)
    for i,(x,y,width,label,lines) in enumerate(nodes):
        draw.rounded_rectangle((x,y,x+width,y+190),radius=18,fill=PANEL,outline=color,width=2)
        svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="190" rx="18" fill="{PANEL}" stroke="{color}" stroke-width="2"/>')
        text(x+22,y+20,label,29,color)
        for j,line in enumerate(lines): text(x+22,y+82+j*41,line,25,MUTED)
        if i<len(nodes)-1:
            nxt=nodes[i+1]
            if nxt[1]==y:
                a,b=(x+width+5,y+95),(nxt[0]-10,y+95)
            else:
                a,b=(x+width/2,y+195),(nxt[0]+nxt[2]/2,nxt[1]-15)
            draw.line((a,b),fill=color,width=4)
            draw.polygon((b,(b[0]-12,b[1]-8),(b[0]-12,b[1]+8)),fill=color)
            svg.append(f'<path d="M{a[0]} {a[1]} L{b[0]} {b[1]}" stroke="{color}" stroke-width="4" marker-end="url(#arrow)"/>')
    if note:
        draw.rounded_rectangle((60,535,1540,675),radius=18,fill=PANEL,outline=AMBER,width=2)
        svg.append(f'<rect x="60" y="535" width="1480" height="140" rx="18" fill="{PANEL}" stroke="{AMBER}" stroke-width="2"/>')
        for j,line in enumerate(note): text(85,555+j*48,line,26,AMBER)
    draw.line((60,810,1540,810),fill=LINE,width=2)
    for j,line in enumerate(footer): text(60,740+j*62,line,25,MUTED)
    image.save(ROOT/(name+'.png'),optimize=True)
    svg.append('</svg>')
    (ROOT/(name+'.svg')).write_text('\n'.join(svg),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/NotoSansSC-VF.ttf'))
    args=parser.parse_args()
    build(args.font,'original-architecture','原有系统架构：消息到分析结果',
          '架构说明图；本轮视频演示不代表 Kafka / S3 / Worker / Triton / Milvus 已整体运行',[
              (60,285,220,'01 Kafka',['关键帧消息','事件入口']),
              (310,285,220,'02 S3',['读取图像','对象存储']),
              (560,285,220,'03 Worker',['视觉编排','并发与背压']),
              (810,285,220,'04 Triton',['GPU 模型','推理服务']),
              (1060,285,220,'05 Milvus',['特征向量','检索与入库']),
              (1310,285,230,'06 结果',['Kafka 输出','识别元数据'])],
          ['顺序用于说明主要职责；对象读写、推理与检索在原代码中由 Worker 编排。',
           '这是原有架构示意，不是本轮实跑证据；完整系统联调状态以各组件日志为准。'],AMBER)
    build(args.font,'identity-flow','已运行：GPU 人脸确认与人形候选匹配',
          'YuNet / SFace / OSNet-AIN 均已实际调用；最终身份由人脸确认，再关联人形',[
              (60,285,260,'01 人脸 ROI',['GPU YuNet 检测','人脸框 + 5点']),
              (365,285,260,'02 人脸对齐',['5点仿射变换','112×112 人脸']),
              (670,285,260,'03 GPU SFace',['真实特征向量','保留执行证据']),
              (975,285,260,'04 比较 / 登记',['cosine 与阈值','匿名身份确认']),
              (1280,285,260,'05 关联人形',['人脸空间绑定','身份状态随帧输出'])],
          ['Source A / B 是剪辑来源标签；Demo-P001 / P002 来自本次实际登记和确认。',
           'A/B/A 是显式人工剪辑；不宣称自然跨镜头跟踪或遮挡后的 ReID 能力。'],GREEN,
          note=['已运行 GPU OSNet-AIN：专用人形特征与候选匹配；人脸负责最终身份确认。',
                '无有效人脸时只保留 body candidate / 未确认；当前样片没有遮挡，不作遮挡能力结论。'])


if __name__=='__main__': main()
