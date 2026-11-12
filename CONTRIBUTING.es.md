# Cómo contribuir a Doehlert Suite

Gracias por considerar una contribución.

## Antes de abrir un pull request

1. Abrí un issue para cambios importantes, así se puede discutir el diseño antes de implementarlo.
2. Mantené explícitos los supuestos estadísticos, fórmulas, restricciones numéricas y unidades.
3. Evitá modificar el motor estadístico y la interfaz en el mismo cambio, salvo que sea necesario.
4. Agregá o actualizá pruebas automáticas para los cambios de comportamiento.
5. Actualizá la documentación en inglés y español cuando cambie el comportamiento visible para el usuario.

## Ejecutar verificaciones

```bash
python -m compileall -q .
python -m unittest discover -s auditor/tests -v
```

## Lista de control del pull request

- [ ] El cambio tiene descripción y justificación claras.
- [ ] Se agregaron o actualizaron pruebas cuando corresponde.
- [ ] La documentación de usuario en inglés y español está sincronizada.
- [ ] No se incluyeron datos experimentales privados, exportaciones generadas, credenciales ni configuraciones locales.
