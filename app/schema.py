# -*- coding: utf-8 -*-
#
# 通用schema
# Author: Xu Chao
# Email: xuchao@illusiontech.cn
# Created Time: 2023-12-14
from pydantic import BaseModel, Field


class MessageResp(BaseModel):
    message: str = Field(..., title='提示信息', description='提示信息')


class VersionResp(BaseModel):
    version: str = Field(..., title='版本信息', description='版本信息')


class StatusCodeResp(BaseModel):
    code: int = Field(..., title='异常状态值', description='异常状态值')
    message: str = Field(..., title='异常状态值说明', description='异常状态值说明')
