# Acceso privado y límites

Configurar RAG_API_TOKEN con al menos 32 caracteres aleatorios en el entorno del servidor. Al iniciar, la interfaz solicita esa clave y la conserva solo durante la pestaña. Sin configuración el API privado responde 503; sin token responde 401. /api/health permanece público. No incluir el token en Git ni en archivos públicos.

El servidor escucha por defecto en 127.0.0.1. Solo exponerlo en otra dirección mediante RAG_HOST detrás de HTTPS y controles de tráfico. RAG_ALLOWED_ORIGINS permite una lista explícita de orígenes separados por coma; no habilitar comodín con credenciales. Las operaciones de consulta, indexación y borrado están protegidas; el borrado no se ejecutó contra datos reales durante las pruebas.

Se limitó query a 4000 caracteres y top_k a 1..20, se ocultaron excepciones internas y se escaparon nombres de fuentes antes de renderizar HTML. Se comprobaron 401, 403 para origen ajeno, 200 autorizado, 422 para límites y 503 sin configuración mediante ASGI y componentes simulados.

Pendiente: actualizar y validar el conjunto de dependencias de IA en un entorno dedicado. OSV detectó avisos en 21 versiones del requirements.txt revisado; este parche de acceso no los resuelve. Añadir límites de concurrencia y consumo del modelo antes de uso público. No se descargaron ni ejecutaron modelos ni se modificó una colección existente.

Con FastAPI/httpx instalados, repetir la prueba de guardas con `python tests/test_security.py`; los componentes RAG son simulados.
