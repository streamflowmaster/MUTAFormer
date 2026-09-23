const $=s=>document.querySelector(s);let mode='full',pollTimer;
// Keep existing callers, but decode all API replies defensively, including
// HTML error pages returned by reverse proxies or older Flask deployments.
const fetch=async(...args)=>{
  let response;
  try{response=await globalThis.fetch(...args)}catch{
    throw new Error('Cannot reach the server. Check the website address and network connection.');
  }
  const decode=async()=>{
    const body=await response.text();
    let data;
    try{data=JSON.parse(body)}catch{
      const hints={413:'The upload is too large. The server limit is 50 MiB for all files combined.',404:'API not found. Check that the frontend and backend versions match.',502:'The proxy cannot reach the application server.',503:'The service is unavailable.',504:'The proxy timed out.',524:'The proxy timed out waiting for the application.'};
      throw new Error(`HTTP ${response.status}: ${hints[response.status]||'The server returned a non-JSON response. Check the server logs or any proxy login page.'}`);
    }
    if(!response.ok)throw new Error(`HTTP ${response.status}: ${data.error||'Request failed.'}`);
    return data;
  };
  return {ok:response.ok,status:response.status,headers:response.headers,json:decode};
};
$('#analysis-protocol').onchange=()=>{const archive=$('#analysis-protocol').value==='paper_archive';for(const id of ['positive-class','epochs','seed','vocab-size','block-size','learning-rate'])$('#'+id).disabled=archive;$('#attr-threshold').readOnly=archive;$('#cluster-distance').readOnly=archive;if(archive){$('#attr-threshold').value='0';$('#cluster-distance').value='15'}$('#protocol-note').textContent=archive?'Requires original RFData in original row order and matching annotations. Reuses the archived attribution (74 nonzero mutations), sequences and blocker input, then recomputes RF screening. No new Transformer training or attribution is claimed.':'10 layers · 70/10/20 patient split · batch 64 · Captum batch 4 · positive class = 0. Original full-cohort vocabulary, class duplication, checkpoint-reference behavior and cross-patient attention are retained. New training does not guarantee the historical 74 mutations.'};
async function checkRuntime(){try{const r=await fetch('/api/health'),d=await r.json();const el=$('#runtime');el.className='runtime '+(d.full_pipeline?'ready':'partial');el.innerHTML=`<span></span>${d.full_pipeline?'Full compute environment ready':'Core environment ready · NUPACK unavailable'}`}catch{$('#runtime').innerHTML='<span></span>Start this tool through web/server.py'}}
document.querySelectorAll('.mode-tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.mode-tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');mode=b.dataset.mode;$('#full-fields').hidden=mode!=='full';$('#dct-fields').hidden=mode!=='dct';showError('')});
for(const id of ['database','annotation','blockers']){const input=$('#'+id),name=$('#'+id+'-name');input.onchange=()=>{if(input.files[0])name.textContent=`${input.files[0].name} · ${(input.files[0].size/1024/1024).toFixed(2)} MB`};const zone=$('#'+id+'-zone');if(zone){['dragenter','dragover'].forEach(e=>zone.addEventListener(e,x=>{x.preventDefault();zone.classList.add('drag')}));['dragleave','drop'].forEach(e=>zone.addEventListener(e,x=>{x.preventDefault();zone.classList.remove('drag')}));zone.addEventListener('drop',e=>{input.files=e.dataTransfer.files;input.dispatchEvent(new Event('change'))})}}
function setRunning(running){$('#run-button').disabled=running;$('#demo-button').disabled=running}
$('#run-form').onsubmit=async e=>{e.preventDefault();if(mode==='full'&&!$('#database').files.length)return showError('Please select a cohort mutation database.');if(mode==='full'&&!$('#annotation').files.length)return showError('Please select a mutation annotation table.');if(mode==='dct'&&!$('#blockers').files.length)return showError('Please select a blocker design table.');showError('');const fd=new FormData(e.target);fd.set('mode',mode);setRunning(true);$('#result-panel').hidden=true;showProgress(4,'Uploading and validating files','Checking column names, data types, and sample structure.');try{const r=await fetch('/api/jobs',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw new Error(d.error||'Unable to start the job.');poll(d.job_id)}catch(err){showError(err.message);setRunning(false);$('#progress-panel').hidden=true}};
$('#demo-button').onclick=async()=>{showError('');setRunning(true);$('#result-panel').hidden=true;showProgress(4,'Starting the example workflow','Loading the built-in cohort and mutation annotation files.');try{const r=await fetch('/api/jobs/demo',{method:'POST'}),d=await r.json();if(!r.ok)throw new Error(d.error||'Unable to start the demo.');poll(d.job_id)}catch(err){showError(err.message);setRunning(false);$('#progress-panel').hidden=true}};
// RFData remains server-side; the separate synthetic demo is still available.
async function runRFData(){
  showError('');setRunning(true);$('#result-panel').hidden=true;
  showProgress(2,'Starting RFData example','Loading server-held data; downloads are disabled.');
  try{
    const r=await fetch('/api/jobs/rfdata',{method:'POST'});
    if(!(r.headers.get('content-type')||'').includes('application/json'))
      throw new Error(`Server returned a non-JSON response (HTTP ${r.status}). Check deployment and server logs.`);
    const d=await r.json();if(!r.ok)throw new Error(d.error||'Unable to start RFData example.');
    poll(d.job_id);
  }catch(err){showError(err.message);setRunning(false);$('#progress-panel').hidden=true;}
}
function updateInputSource(){
  const demo=mode==='full'&&$('#input-source').value==='demo';
  $('#upload-fields').hidden=$('#input-source').value==='demo';
  $('#training-fields').hidden=$('#input-source').value==='demo';
  $('#dataset-note').hidden=$('#input-source').value!=='demo';
  $('.filter-settings').hidden=demo;
  $('#demo-button').hidden=demo;
  // Hidden controls still participate in native validation unless disabled.
  for(const container of ['#upload-fields','#training-fields']){
    document.querySelectorAll(container+' input, '+container+' select').forEach(el=>{
      el.disabled=mode!=='full'||demo;
    });
  }
  $('#blockers').disabled=mode!=='dct';
  document.querySelectorAll('.filter-settings input').forEach(el=>el.disabled=demo);
  if(mode==='full'&&!demo)$('#analysis-protocol').onchange();
  $('#run-button span').textContent=demo?'Run with RFData example':'Run panel design';
  showError('');
}
const submitUploadedData=$('#run-form').onsubmit;
$('#run-form').onsubmit=e=>{
  if(mode==='full'&&$('#input-source').value==='demo'){
    e.preventDefault();
    if(!$('#run-button').disabled)runRFData();
    return;
  }
  return submitUploadedData(e);
};
$('#input-source').onchange=updateInputSource;
document.querySelectorAll('.mode-tab').forEach(b=>b.addEventListener('click',updateInputSource));
updateInputSource();
function showError(t){const e=$('#form-error');e.hidden=!t;e.textContent=t}
function showProgress(p,title,detail){$('#progress-panel').hidden=false;$('#progress-percent').textContent=p+'%';$('#progress-bar').style.width=p+'%';$('#progress-title').textContent=title;$('#progress-detail').textContent=detail;const idx=p<8?0:p<42?1:p<62?2:p<84?3:4;document.querySelectorAll('#step-list li').forEach((x,i)=>x.classList.toggle('active',i<=idx))}
async function poll(id){clearTimeout(pollTimer);try{const r=await fetch('/api/jobs/'+id),d=await r.json();if(!r.ok)throw new Error(d.error||'Unable to read the job.');showProgress(d.progress,d.stage,d.detail);if(d.status==='done'||d.status==='partial'){renderResult(d);setRunning(false);return}if(d.status==='failed'){throw new Error(d.error)}pollTimer=setTimeout(()=>poll(id),1200)}catch(err){showError(err.message);setRunning(false);$('#progress-panel').hidden=true}}
function renderResult(d){showProgress(100,d.status==='partial'?'Available stages completed':'Panel design completed',d.detail);const result=d.result||{},metrics=result.metrics||{};$('#metrics').innerHTML=Object.entries(metrics).map(([k,v])=>`<div class="metric"><span>${k}</span><b>${v}</b></div>`).join('');$('#result-body').innerHTML=(result.rows||[]).map((r,i)=>`<tr><td>${i+1}</td><td>${escapeHtml(r.name||'—')}</td><td>${r.mut_count??'—'}</td><td>${fmt(r.probability)}</td><td>${fmt(r.frequency,0)}</td><td>${fmt(r.attribution,4)}</td><td class="${r.pass?'pass':'fail'}">${r.pass?'Selected':'Excluded'}</td></tr>`).join('')||'<tr><td colspan="7">No recommended candidates are available at this stage.</td></tr>';$('#downloads').innerHTML=(result.files||[]).map(f=>`<a href="${f.url}">${escapeHtml(f.label)} ↓</a>`).join('');const warns=result.warnings||[];$('#warnings').hidden=!warns.length;$('#warnings').innerHTML=warns.map(w=>`<p>${escapeHtml(w)}</p>`).join('');$('#result-panel').hidden=false;$('#result-panel').scrollIntoView({behavior:'smooth',block:'start'})}
function fmt(v,n=3){return v===null||v===undefined||Number.isNaN(Number(v))?'—':Number(v).toFixed(n)}function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}checkRuntime();
