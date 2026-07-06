"""Trace 可视化 HTTP 服务器（标准库，无额外依赖）。

在原版基础上增加了端口配置与 path 安全处理。
"""

import http.server
import json
import os
import threading
from pathlib import Path


def start_trace_server(host="127.0.0.1", port=8765, trace_dir: str = ".trace"):
    """在独立守护线程中启动一个 HTTP 服务，
    提供 trace.json 的轮询接口与简单的 HTML 面板页面。"""
    trace_dir = Path(trace_dir).resolve()

    class TraceHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # 用标准日志或静默

        def _respond_json(self, data, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def _respond_html(self, html, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

        def _fallback_path(self, p: str) -> str:
            """防止路径穿越，只允许 trace_dir 内的文件。"""
            try:
                target = (trace_dir / p).resolve()
                target.relative_to(trace_dir)
                return str(target)
            except (ValueError, RuntimeError):
                return str(trace_dir / "trace.json")

        def do_GET(self):
            if self.path == "/":
                self._respond_html(HTML_PAGE)
            elif self.path == "/api/trace":
                path = self._fallback_path("trace.json")
                try:
                    data = json.loads(Path(path).read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {"error": "no trace data"}
                self._respond_json(data)
            else:
                self.send_response(404)
                self.end_headers()

    server = http.server.HTTPServer((host, port), TraceHandler)
    server.timeout = 1
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"  \033[36m[trace server] http://{host}:{port}\033[0m")
    return server


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>小智 Trace</title>
<style>
  body{font:14px/1.4 monospace;margin:20px;background:#111;color:#ddd}
  .node{margin:4px 0 4px 20px;border-left:2px solid #333;padding-left:10px}
  .node-head{cursor:pointer;display:flex;gap:8px;align-items:center;padding:2px 4px;border-radius:2px}
  .node-head:hover{background:#222}
  .label{color:#eee}.type{color:#888;font-size:11px;min-width:60px}
  .duration{color:#666;font-size:11px;min-width:50px}
  .status_ok{color:#4caf50}.status_error{color:#f44336}.status_running{color:#ff9800}
  .detail{color:#999;font-size:12px;margin:2px 0 2px 90px;white-space:pre-wrap}
  .messages{display:none;margin:4px 0 4px 90px;background:#1a1a1a;padding:8px;border-radius:4px}
  .messages pre{color:#bbb;font-size:11px;margin:2px 0;white-space:pre-wrap;word-break:break-all}
  .tree-toggle{display:flex;gap:12px;margin:10px 0}
  button{background:#333;color:#ddd;border:1px solid #555;padding:4px 12px;border-radius:4px;cursor:pointer}
  button:hover{background:#444}
  .stats{color:#888;font-size:12px;margin:10px 0}
</style></head>
<body>
<h2>小智 Trace</h2>
<div class="stats" id="stats"></div>
<div class="tree-toggle">
  <button onclick="expandAll(true)">展开全部</button>
  <button onclick="expandAll(false)">折叠全部</button>
  <button onclick="autoExpand()">仅展开运行中</button>
</div>
<div id="tree"></div>
<div id="messages"></div>
<script>
let data=null;
async function poll(){try{
  let r=await fetch('/api/trace');
  data=await r.json();
  render(data);
}catch(e){setTimeout(poll,1000)}}
function render(d){
  const tree=document.getElementById('tree');
  const stats=document.getElementById('stats');
  let total=0,nodes=countNodes(d.roots||[]);
  stats.textContent=`树节点:${nodes} 消息:${(d.messages||[]).length}`;
  tree.innerHTML=d.roots?d.roots.map(n=>renderNode(n,0)).join(''):'<em>暂无 trace 数据</em>';
  document.getElementById('messages').innerHTML=
    (d.messages||[]).map(m=>`<div class="messages" style="display:block"><pre>${esc(JSON.stringify(m,null,2))}</pre></div>`).join('');
}
function countNodes(roots){let c=0;for(let r of roots){c++;for(let ch of r.children||[])c+=countChildren(ch)}return c}
function countChildren(n){let c=1;for(let ch of n.children||[])c+=countChildren(ch);return c}
function renderNode(n,depth){
  let ms=n.duration?((n.duration*1000).toFixed(0)+'ms'):'—';
  let cls='status_'+(n.status||'running');
  let sub=(n.children||[]).map(ch=>renderNode(ch,depth+1)).join('');
  return `<div class="node">
    <div class="node-head" onclick="toggle(this)">
      <span class="type">${esc(n.type||'')}</span>
      <span class="label">${esc(n.label||'')}</span>
      <span class="duration">${ms}</span>
      <span class="${cls}">${n.status||'running'}</span>
    </div>
    ${n.detail?`<div class="detail">${esc(n.detail)}</div>`:''}
    ${sub?`<div class="children" style="display:${depth<2?'block':'none'}">${sub}</div>`:''}
  </div>`;
}
function toggle(el){const ch=el.nextElementSibling;if(ch&&ch.classList.contains('children'))ch.style.display=ch.style.display=='none'?'block':'none';}
function expandAll(v){
  document.querySelectorAll('.children').forEach(c=>c.style.display=v?'block':'none');
  document.querySelectorAll('.messages').forEach(m=>m.style.display=v?'block':'none');
}
function autoExpand(){
  document.querySelectorAll('.children').forEach(c=>{
    const hasRunning=c.querySelector('.status_running');
    c.style.display=hasRunning?'block':'none';
  });
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
setInterval(poll,500);poll();
</script></body></html>"""