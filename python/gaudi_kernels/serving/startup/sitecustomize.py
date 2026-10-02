"""Transitional worker initialization; removed by the vLLM factory phase."""
from gaudi_kernels.serving.bootstrap import start
try:
    start()
except BaseException as error:
    raise SystemExit('Canonical bootstrap rejected: ' + str(error))
