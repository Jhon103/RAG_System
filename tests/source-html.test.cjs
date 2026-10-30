const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

test('las fuentes no generan etiquetas HTML ni colisionan con el prototipo',()=>{
 const elements=new Map();
 const document={addEventListener(){},getElementById(id){if(!elements.has(id))elements.set(id,{classList:{add(){}},scrollIntoView(){}});return elements.get(id);},createElement(){return {set textContent(value){this.value=String(value);},get innerHTML(){return this.value.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}};}};
 const context=vm.createContext({document,console});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8'),context);
 for(const source of ['<img src=x onerror=alert(1)>','__proto__']){
  context.fixture={query:'test',confidence:1,elapsed_ms:1,chunks_used:1,sources:[source],avg_similarity:1,results:[{source,text:'test',similarity:1,rank:1}]};
  vm.runInContext('renderResults(fixture)',context);
  assert.ok(!elements.get('sourcesList').innerHTML.includes('<img'));
  assert.ok(!elements.get('chunksGrid').innerHTML.includes('<img'));
 }
});
