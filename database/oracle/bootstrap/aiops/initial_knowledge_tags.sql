-- AIOps 运维知识标签初始词表（Oracle）。
--
-- 使用方式（SQL*Plus / SQLcl / SQL Developer Run Script）：
--   1. 将 domain_name 修改为目标 AIOps Domain 名称；
--   2. 执行本脚本。
--
-- 脚本只补充缺失标签，不覆盖已存在标签的名称、说明、状态或排序。

DEFINE domain_name = aiops_portal

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_domain_id KBOT_PLATFORM_DOMAIN.DOMAIN_ID%TYPE;
    l_table_count PLS_INTEGER;
    l_duplicate_count PLS_INTEGER;
    l_inserted_count PLS_INTEGER := 0;
    l_normalized_name KBOT_OPS_KNOWLEDGE_TAG.NORMALIZED_NAME%TYPE;
    l_description KBOT_OPS_KNOWLEDGE_TAG.DESCRIPTION%TYPE;
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME IN ('KBOT_PLATFORM_DOMAIN', 'KBOT_OPS_KNOWLEDGE_TAG');

    IF l_table_count <> 2 THEN
        RAISE_APPLICATION_ERROR(
            -20001,
            '缺少 KBOT_PLATFORM_DOMAIN 或 KBOT_OPS_KNOWLEDGE_TAG，请先完成 AIOps Schema 初始化。'
        );
    END IF;

    SELECT DOMAIN_ID
      INTO l_domain_id
      FROM KBOT_PLATFORM_DOMAIN
     WHERE NAME = '&domain_name'
       AND STATUS = 'ACTIVE';

    SELECT COUNT(*)
      INTO l_duplicate_count
      FROM (
          SELECT CATEGORY, NORMALIZED_NAME
            FROM KBOT_OPS_KNOWLEDGE_TAG
           WHERE DOMAIN_ID = l_domain_id
           GROUP BY CATEGORY, NORMALIZED_NAME
          HAVING COUNT(*) > 1
      );

    IF l_duplicate_count > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20002,
            '目标 Domain 已存在重复知识标签，请先完成治理后再执行初始词表。'
        );
    END IF;

    FOR item IN (
        SELECT category, display_name, sort_order
          FROM JSON_TABLE(
              q'~[
                {"category":"RESOURCE_CATEGORY","display_name":"物理服务器","sort_order":10},
                {"category":"RESOURCE_CATEGORY","display_name":"虚拟机","sort_order":20},
                {"category":"RESOURCE_CATEGORY","display_name":"云主机","sort_order":30},
                {"category":"RESOURCE_CATEGORY","display_name":"容器","sort_order":40},
                {"category":"RESOURCE_CATEGORY","display_name":"Kubernetes 集群","sort_order":50},
                {"category":"RESOURCE_CATEGORY","display_name":"关系型数据库","sort_order":60},
                {"category":"RESOURCE_CATEGORY","display_name":"NoSQL 数据库","sort_order":70},
                {"category":"RESOURCE_CATEGORY","display_name":"缓存服务","sort_order":80},
                {"category":"RESOURCE_CATEGORY","display_name":"搜索与日志引擎","sort_order":90},
                {"category":"RESOURCE_CATEGORY","display_name":"数据仓库与大数据平台","sort_order":100},
                {"category":"RESOURCE_CATEGORY","display_name":"网络设备","sort_order":110},
                {"category":"RESOURCE_CATEGORY","display_name":"负载均衡","sort_order":120},
                {"category":"RESOURCE_CATEGORY","display_name":"DNS 与 CDN","sort_order":130},
                {"category":"RESOURCE_CATEGORY","display_name":"防火墙与安全设备","sort_order":140},
                {"category":"RESOURCE_CATEGORY","display_name":"Web 服务器","sort_order":150},
                {"category":"RESOURCE_CATEGORY","display_name":"应用服务器","sort_order":160},
                {"category":"RESOURCE_CATEGORY","display_name":"消息队列","sort_order":170},
                {"category":"RESOURCE_CATEGORY","display_name":"API 网关","sort_order":180},
                {"category":"RESOURCE_CATEGORY","display_name":"存储系统","sort_order":190},
                {"category":"RESOURCE_CATEGORY","display_name":"备份与恢复系统","sort_order":200},
                {"category":"RESOURCE_CATEGORY","display_name":"监控与可观测平台","sort_order":210},
                {"category":"RESOURCE_CATEGORY","display_name":"CI/CD 平台","sort_order":220},
                {"category":"RESOURCE_CATEGORY","display_name":"身份与访问管理","sort_order":230},
                {"category":"RESOURCE_CATEGORY","display_name":"配置中心与服务注册","sort_order":240},
                {"category":"RESOURCE_CATEGORY","display_name":"业务应用","sort_order":250}
              ]~',
              '$[*]' COLUMNS (
                  category VARCHAR2(32) PATH '$.category',
                  display_name VARCHAR2(128) PATH '$.display_name',
                  sort_order NUMBER PATH '$.sort_order'
              )
          )
        UNION ALL
        SELECT category, display_name, sort_order
          FROM JSON_TABLE(
              q'~[
                {"category":"TECHNOLOGY_STACK","display_name":"Linux","sort_order":10},
                {"category":"TECHNOLOGY_STACK","display_name":"Windows Server","sort_order":20},
                {"category":"TECHNOLOGY_STACK","display_name":"IBM AIX","sort_order":30},
                {"category":"TECHNOLOGY_STACK","display_name":"Oracle Solaris","sort_order":40},
                {"category":"TECHNOLOGY_STACK","display_name":"VMware vSphere","sort_order":50},
                {"category":"TECHNOLOGY_STACK","display_name":"KVM","sort_order":60},
                {"category":"TECHNOLOGY_STACK","display_name":"Microsoft Hyper-V","sort_order":70},
                {"category":"TECHNOLOGY_STACK","display_name":"Docker","sort_order":80},
                {"category":"TECHNOLOGY_STACK","display_name":"containerd","sort_order":90},
                {"category":"TECHNOLOGY_STACK","display_name":"Kubernetes","sort_order":100},
                {"category":"TECHNOLOGY_STACK","display_name":"Red Hat OpenShift","sort_order":110},
                {"category":"TECHNOLOGY_STACK","display_name":"Helm","sort_order":120},
                {"category":"TECHNOLOGY_STACK","display_name":"Oracle Database","sort_order":130},
                {"category":"TECHNOLOGY_STACK","display_name":"Oracle RAC","sort_order":140},
                {"category":"TECHNOLOGY_STACK","display_name":"Oracle Data Guard","sort_order":150},
                {"category":"TECHNOLOGY_STACK","display_name":"PostgreSQL","sort_order":160},
                {"category":"TECHNOLOGY_STACK","display_name":"Patroni","sort_order":170},
                {"category":"TECHNOLOGY_STACK","display_name":"MySQL","sort_order":180},
                {"category":"TECHNOLOGY_STACK","display_name":"MariaDB","sort_order":190},
                {"category":"TECHNOLOGY_STACK","display_name":"Microsoft SQL Server","sort_order":200},
                {"category":"TECHNOLOGY_STACK","display_name":"IBM Db2","sort_order":210},
                {"category":"TECHNOLOGY_STACK","display_name":"MongoDB","sort_order":220},
                {"category":"TECHNOLOGY_STACK","display_name":"Redis","sort_order":230},
                {"category":"TECHNOLOGY_STACK","display_name":"NGINX","sort_order":240},
                {"category":"TECHNOLOGY_STACK","display_name":"Apache HTTP Server","sort_order":250},
                {"category":"TECHNOLOGY_STACK","display_name":"Microsoft IIS","sort_order":260},
                {"category":"TECHNOLOGY_STACK","display_name":"Apache Tomcat","sort_order":270},
                {"category":"TECHNOLOGY_STACK","display_name":"Oracle WebLogic","sort_order":280},
                {"category":"TECHNOLOGY_STACK","display_name":"IBM WebSphere","sort_order":290},
                {"category":"TECHNOLOGY_STACK","display_name":"Red Hat JBoss EAP","sort_order":300}
              ]~',
              '$[*]' COLUMNS (
                  category VARCHAR2(32) PATH '$.category',
                  display_name VARCHAR2(128) PATH '$.display_name',
                  sort_order NUMBER PATH '$.sort_order'
              )
          )
        UNION ALL
        SELECT category, display_name, sort_order
          FROM JSON_TABLE(
              q'~[
                {"category":"TECHNOLOGY_STACK","display_name":"Apache Kafka","sort_order":310},
                {"category":"TECHNOLOGY_STACK","display_name":"RabbitMQ","sort_order":320},
                {"category":"TECHNOLOGY_STACK","display_name":"Apache RocketMQ","sort_order":330},
                {"category":"TECHNOLOGY_STACK","display_name":"IBM MQ","sort_order":340},
                {"category":"TECHNOLOGY_STACK","display_name":"Elasticsearch","sort_order":350},
                {"category":"TECHNOLOGY_STACK","display_name":"OpenSearch","sort_order":360},
                {"category":"TECHNOLOGY_STACK","display_name":"Apache Hadoop","sort_order":370},
                {"category":"TECHNOLOGY_STACK","display_name":"Apache Spark","sort_order":380},
                {"category":"TECHNOLOGY_STACK","display_name":"Prometheus","sort_order":390},
                {"category":"TECHNOLOGY_STACK","display_name":"Grafana","sort_order":400},
                {"category":"TECHNOLOGY_STACK","display_name":"Zabbix","sort_order":410},
                {"category":"TECHNOLOGY_STACK","display_name":"Nagios","sort_order":420},
                {"category":"TECHNOLOGY_STACK","display_name":"Elastic Stack","sort_order":430},
                {"category":"TECHNOLOGY_STACK","display_name":"Grafana Loki","sort_order":440},
                {"category":"TECHNOLOGY_STACK","display_name":"OpenTelemetry","sort_order":450},
                {"category":"TECHNOLOGY_STACK","display_name":"Ansible","sort_order":460},
                {"category":"TECHNOLOGY_STACK","display_name":"Terraform","sort_order":470},
                {"category":"TECHNOLOGY_STACK","display_name":"Jenkins","sort_order":480},
                {"category":"TECHNOLOGY_STACK","display_name":"GitLab CI/CD","sort_order":490},
                {"category":"TECHNOLOGY_STACK","display_name":"Argo CD","sort_order":500},
                {"category":"TECHNOLOGY_STACK","display_name":"Ceph","sort_order":510},
                {"category":"TECHNOLOGY_STACK","display_name":"MinIO","sort_order":520},
                {"category":"TECHNOLOGY_STACK","display_name":"HAProxy","sort_order":530},
                {"category":"TECHNOLOGY_STACK","display_name":"TiDB","sort_order":540},
                {"category":"TECHNOLOGY_STACK","display_name":"OceanBase Database","sort_order":550},
                {"category":"TECHNOLOGY_STACK","display_name":"Dameng Database","sort_order":560},
                {"category":"TECHNOLOGY_STACK","display_name":"KingbaseES","sort_order":570},
                {"category":"TECHNOLOGY_STACK","display_name":"etcd","sort_order":580},
                {"category":"TECHNOLOGY_STACK","display_name":"Nacos","sort_order":590},
                {"category":"TECHNOLOGY_STACK","display_name":"HashiCorp Consul","sort_order":600},
                {"category":"TECHNOLOGY_STACK","display_name":"Istio","sort_order":610},
                {"category":"TECHNOLOGY_STACK","display_name":"Oracle RMAN","sort_order":620},
                {"category":"TECHNOLOGY_STACK","display_name":"pgBackRest","sort_order":630},
                {"category":"TECHNOLOGY_STACK","display_name":"Veeam Backup & Replication","sort_order":640}
              ]~',
              '$[*]' COLUMNS (
                  category VARCHAR2(32) PATH '$.category',
                  display_name VARCHAR2(128) PATH '$.display_name',
                  sort_order NUMBER PATH '$.sort_order'
              )
          )
        UNION ALL
        SELECT category, display_name, sort_order
          FROM JSON_TABLE(
              q'~[
                {"category":"VENDOR","display_name":"Oracle","sort_order":10},
                {"category":"VENDOR","display_name":"Microsoft","sort_order":20},
                {"category":"VENDOR","display_name":"IBM","sort_order":30},
                {"category":"VENDOR","display_name":"Red Hat","sort_order":40},
                {"category":"VENDOR","display_name":"SUSE","sort_order":50},
                {"category":"VENDOR","display_name":"Canonical","sort_order":60},
                {"category":"VENDOR","display_name":"VMware","sort_order":70},
                {"category":"VENDOR","display_name":"Amazon Web Services","sort_order":80},
                {"category":"VENDOR","display_name":"Google Cloud","sort_order":90},
                {"category":"VENDOR","display_name":"Alibaba Cloud","sort_order":100},
                {"category":"VENDOR","display_name":"Huawei Cloud","sort_order":110},
                {"category":"VENDOR","display_name":"Tencent Cloud","sort_order":120},
                {"category":"VENDOR","display_name":"Cisco","sort_order":130},
                {"category":"VENDOR","display_name":"Huawei","sort_order":140},
                {"category":"VENDOR","display_name":"H3C","sort_order":150},
                {"category":"VENDOR","display_name":"Juniper Networks","sort_order":160},
                {"category":"VENDOR","display_name":"F5","sort_order":170},
                {"category":"VENDOR","display_name":"Fortinet","sort_order":180},
                {"category":"VENDOR","display_name":"Palo Alto Networks","sort_order":190},
                {"category":"VENDOR","display_name":"Dell Technologies","sort_order":200},
                {"category":"VENDOR","display_name":"Hewlett Packard Enterprise","sort_order":210},
                {"category":"VENDOR","display_name":"Lenovo","sort_order":220},
                {"category":"VENDOR","display_name":"Inspur","sort_order":230},
                {"category":"VENDOR","display_name":"NetApp","sort_order":240},
                {"category":"VENDOR","display_name":"Nutanix","sort_order":250},
                {"category":"VENDOR","display_name":"Veeam","sort_order":260},
                {"category":"VENDOR","display_name":"Veritas","sort_order":270},
                {"category":"VENDOR","display_name":"Commvault","sort_order":280},
                {"category":"VENDOR","display_name":"SAP","sort_order":290},
                {"category":"VENDOR","display_name":"MongoDB","sort_order":300},
                {"category":"VENDOR","display_name":"Redis","sort_order":310},
                {"category":"VENDOR","display_name":"Elastic","sort_order":320},
                {"category":"VENDOR","display_name":"Grafana Labs","sort_order":330},
                {"category":"VENDOR","display_name":"Docker","sort_order":340},
                {"category":"VENDOR","display_name":"HashiCorp","sort_order":350},
                {"category":"VENDOR","display_name":"GitLab","sort_order":360},
                {"category":"VENDOR","display_name":"Atlassian","sort_order":370},
                {"category":"VENDOR","display_name":"Apache Software Foundation","sort_order":380},
                {"category":"VENDOR","display_name":"Cloud Native Computing Foundation","sort_order":390},
                {"category":"VENDOR","display_name":"PingCAP","sort_order":400},
                {"category":"VENDOR","display_name":"OceanBase","sort_order":410},
                {"category":"VENDOR","display_name":"Dameng","sort_order":420},
                {"category":"VENDOR","display_name":"Kingbase","sort_order":430},
                {"category":"VENDOR","display_name":"Sangfor","sort_order":440}
              ]~',
              '$[*]' COLUMNS (
                  category VARCHAR2(32) PATH '$.category',
                  display_name VARCHAR2(128) PATH '$.display_name',
                  sort_order NUMBER PATH '$.sort_order'
              )
          )
    ) LOOP
        l_normalized_name := UPPER(
            REGEXP_REPLACE(TRIM(item.display_name), '[[:space:]]+', ' ')
        );
        l_description := CASE item.category
            WHEN 'RESOURCE_CATEGORY' THEN
                item.display_name || '类资源的部署、配置、监控与故障处理知识。'
            WHEN 'TECHNOLOGY_STACK' THEN
                item.display_name || '技术栈的安装、配置、升级、监控与故障处理知识。'
            WHEN 'VENDOR' THEN
                item.display_name || '产品与服务相关的运维知识。'
        END;

        MERGE INTO KBOT_OPS_KNOWLEDGE_TAG target
        USING (
            SELECT
                l_domain_id AS domain_id,
                item.category AS category,
                item.display_name AS display_name,
                l_normalized_name AS normalized_name,
                l_description AS description,
                item.sort_order AS sort_order
            FROM DUAL
        ) source
        ON (
            target.DOMAIN_ID = source.domain_id
            AND target.CATEGORY = source.category
            AND target.NORMALIZED_NAME = source.normalized_name
        )
        WHEN NOT MATCHED THEN INSERT (
            TAG_ID, DOMAIN_ID, CATEGORY, DISPLAY_NAME, NORMALIZED_NAME,
            DESCRIPTION, STATUS, SORT_ORDER, CREATED_BY, UPDATED_BY,
            CREATED_AT, UPDATED_AT, ROW_VERSION
        ) VALUES (
            SYS_GUID(), source.domain_id, source.category, source.display_name,
            source.normalized_name, source.description, 'ACTIVE', source.sort_order,
            'bootstrap:aiops_knowledge_tags', 'bootstrap:aiops_knowledge_tags',
            SYSTIMESTAMP, SYSTIMESTAMP, 1
        );

        l_inserted_count := l_inserted_count + SQL%ROWCOUNT;
    END LOOP;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps 知识标签初始化完成：Domain=&domain_name，新增=' || l_inserted_count
    );
END;
/

COMMIT;

SELECT CATEGORY, COUNT(*) AS ACTIVE_TAG_COUNT
  FROM KBOT_OPS_KNOWLEDGE_TAG
 WHERE DOMAIN_ID = (
     SELECT DOMAIN_ID FROM KBOT_PLATFORM_DOMAIN WHERE NAME = '&domain_name'
 )
   AND STATUS = 'ACTIVE'
 GROUP BY CATEGORY
 ORDER BY CATEGORY;
