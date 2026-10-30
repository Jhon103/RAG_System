import os, sys, types, importlib.util, asyncio, json
from pathlib import Path
for module, cls in [('document_loader','DocumentLoader'),('chunker','TextChunker'),('vector_store','VectorStore'),('retriever','Retriever'),('generator','ResponseGenerator')]:
    stub=types.ModuleType(module);setattr(stub,cls,type(cls,(),{}));sys.modules[module]=stub
os.environ['RAG_API_TOKEN']='test-only-'+'x'*40
spec=importlib.util.spec_from_file_location('audit_rag',str(Path(__file__).resolve().parents[1] / 'web_server.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
import httpx
async def main():
    writes=[]
    class VectorStore:
        def delete_collection(self):writes.append(True)
    module.rag_components['vector_store']=VectorStore()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app),base_url='http://testserver') as client:
        assert (await client.post('/api/clear')).status_code==401
        assert not writes
        headers={'Authorization':'Bearer '+os.environ['RAG_API_TOKEN']}
        assert (await client.post('/api/clear',headers={**headers,'Origin':'https://evil.invalid'})).status_code==403
        assert not writes
        assert (await client.post('/api/clear',headers=headers)).status_code==200
        assert len(writes)==1
        assert (await client.post('/api/query',headers=headers,json={'query':'x','top_k':5000})).status_code==422
        assert (await client.post('/api/query',headers=headers,json={'query':'x'*4001})).status_code==422
        module.API_TOKEN=''
        assert (await client.post('/api/clear',headers=headers)).status_code==503
    result={'unauthenticated':401,'foreignOrigin':403,'authenticated':200,'bounds':422,'missingSecret':503,'externalDatabase':False}
    print(json.dumps(result))
asyncio.run(main())
