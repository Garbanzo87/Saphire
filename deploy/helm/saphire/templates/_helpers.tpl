{{- define "saphire.envFrom" -}}
envFrom:
  - secretRef:
      name: {{ .Values.secrets.name }}
env:
{{- range $k, $v := .Values.env }}
  - name: {{ $k }}
    value: {{ $v | quote }}
{{- end }}
{{- end -}}
{{- define "saphire.volumes" -}}
{{- if .Values.artifacts.pvc.enabled }}
volumeMounts:
  - { name: artifacts, mountPath: /artifacts }
{{- end }}
{{- end -}}
{{- define "saphire.podVolumes" -}}
{{- if .Values.artifacts.pvc.enabled }}
volumes:
  - name: artifacts
    persistentVolumeClaim: { claimName: saphire-artifacts }
{{- end }}
{{- end -}}
