# test



## 接口使用说明

### 统一的接口异常响应值

```json
{
    "code": 10404,
    "message": "异常响应信息",
    "detail": "详细的接口异常信息，通常只用于开发阶段排查定位问题，该值可能为空"
}
```

- code值和http状态码通常是对应的，如该值为10404时，则http状态对应为404（计算方式：10404 % 1000），如果计算得到的http状态码大于600，则会重置为500

### 基础通用接口

- `/version`: 获取接口版本号
- `/status/code`: 获取接口异常状态码列表